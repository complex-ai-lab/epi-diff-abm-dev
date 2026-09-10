import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import re
import math
import glob
import pandas as pd
import numpy as np
import os

from agent_torch.core.substep import SubstepTransitionMessagePassing
from agent_torch.core.helpers import get_by_path
from agent_torch.core.helpers import set_by_path
from agent_torch.core.distributions import StraightThroughBernoulli

# --- Age encoding for this model ---------------------------------------------
# populations/pop<FIPS>/age.pickle stores a 6-value code per agent
# (scripts/process_counties.py::create_county_folder, populations/*/mapping.json):
#   0 = 20t29   1 = 30t39   2 = 40t49   3 = 50t64   4 = 65A (65+)   5 = U19 (<20)
# The child / adult / elderly split used by the *diagnostic* age-proportion and
# age-incidence outputs. NOTE: get_mean_agent_interactions() below deliberately
# does NOT use this — its (mis-labelled) split is baked into every existing
# calibration and re-labelling it would change the factual dynamics.
CHILD_CODES = (5,)              # U19
ADULT_CODES = (0, 1, 2, 3)     # 20-64
ELDERLY_CODES = (4,)           # 65A
VALID_AGE_CODES = frozenset(CHILD_CODES + ADULT_CODES + ELDERLY_CODES)

# --- Age-priority vaccination CF family (CF 12-17) ---------------------------
# {age_code: rank}; higher rank is vaccinated sooner. Ties within a rank stay
# Gumbel-random so the CRN comparison across CF types is still meaningful.
AGE_PRIORITY = {
    "elderly_first":     {4: 5, 3: 4, 2: 3, 1: 2, 0: 1, 5: 0},
    "working_age_first": {0: 2, 1: 2, 2: 2, 3: 2, 4: 1, 5: 0},
}
# CF type -> AGE_PRIORITY ordering. The school/work levers for these types are
# set in `logic_map` inside forward().
CF_VACC_PRIORITY = {
    12: "elderly_first",     13: "working_age_first",
    14: "working_age_first", 15: "working_age_first",
    16: "elderly_first",     17: "elderly_first",
}

# --- Work-essentiality CF family (CF 18-23) ---------------------------------
# CISA v4.0 "Essential Critical Infrastructure Workforce" -> this repo's 19
# NAICS sectors. EDUCATION is essential for vaccine priority and exempt from the
# non-essential occ closure (double-controlled with the school lever). FOOD is
# essential by default (grocery/production); flipping it is the sensitivity axis.
ESSENTIAL_SECTORS = frozenset({
    "AGRICULTURE", "MINING", "UTILITIES", "TRANSPORTATION", "HEALTHCARE", "FOOD",
    "MANUFACTURING", "CONSTRUCTION", "WASTEMANAGEMENT", "OTHER", "EDUCATION",
})
NON_ESSENTIAL_SECTORS = frozenset({
    "WHOLESALETRADE", "RETAILTRADE", "INFORMATION", "FINANCEINSURANCE",
    "REALESTATERENTAL", "SCIENTIFICTECHNICAL", "ENTERPRISEMANAGEMENT", "ART",
})
ESSENTIALITY_CF_TYPES = frozenset({18, 19, 20, 21, 22, 23, 24})
CF_ESSENTIAL_VACC = frozenset({18, 20, 21, 23})   # essential-first vaccine
CF_SELECTIVE_OCC = frozenset({19, 20, 22, 23})    # non-essential-only closure
# CF 24 = ALL-workplace closure, factual school -- the full-closure reference for
# retained_benefit = (CF7 - CF19) / (CF7 - CF24).
# essentiality group code: 0 = not in workforce, 1 = essential, 2 = non-essential

# --- Household-level allocation CF family (CF 25-28) ------------------------
# Whole-household dose selection: once a household is reached in the campaign
# order, every eligible (susceptible) member is vaccinated before the next
# household. Daily dose count = factual vaccines(t); NPIs factual for all four.
# Reference = CF 7 (individual, uniform). Runs on covid_output_causal +
# log1p_rw2_l5_rmax5 (no netgen, no recalibration -- the lever only reorders
# who gets a dose, never contact structure).
HH_ALLOC_CF_TYPES = frozenset({25, 26, 27, 28})
CF_HH_ORDER = {
    25: "random",               # random household order
    26: "elderly_home_first",   # households with >=1 elderly member first
    27: "largest_first",        # largest households first
    28: "essential_home_first", # households with >=1 essential worker first
}


class NewTransmission(SubstepTransitionMessagePassing):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]
        self.RECOVERED_VAR = self.config["simulation_metadata"]["RECOVERED_VAR"]
        self.INFECTED_VAR = self.config["simulation_metadata"]["INFECTED_VAR"]
        self.MORTALITY_VAR = self.config["simulation_metadata"]["MORTALITY_VAR"]
        # Distinct disease stage for vaccine-derived protection (S -> VACCINATED
        # -> S). Kept separate from RECOVERED_VAR so the two immunities can wane
        # independently. SFInfector in config.yaml is sized to cover this id.
        self.VACCINATED_VAR = self.config["simulation_metadata"].get(
            "VACCINATED_VAR", 5
        )

        self.num_timesteps = self.config["simulation_metadata"]["num_steps_per_episode"]
        self.num_weeks = self.config["simulation_metadata"]["NUM_WEEKS"]

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.data_dir = os.path.join(project_root, 'data')

        from dotenv import load_dotenv
        load_dotenv(os.path.join(project_root, '.env'))
        networks_dir_env = os.getenv("NETWORKS_DIR")
        if networks_dir_env:
            if not os.path.isabs(networks_dir_env):
                self.networks_dir = os.path.join(project_root, networks_dir_env)
            else:
                self.networks_dir = networks_dir_env
        else:
            self.networks_dir = os.path.join(self.data_dir, 'networks', 'covid_output_causal')

        self.networks = self._preload_all_networks()
        self.household_net = self._load_single_net(f"{self.networks_dir}/{self.config['simulation_metadata']['POPULATION']}/mobility_networks/HOUSEHOLD_NETWORK.pkl")
        self.school_net = self._load_single_net(f"{self.networks_dir}/{self.config['simulation_metadata']['POPULATION']}/mobility_networks/SCHOOL_NETWORK.pkl")
        self.proportion_history = []
        self.age_proportion_history = []
        # Parallel to proportion_history: [t, vaccinated_fraction]. Kept as a
        # separate list so proportion_history stays 5-wide (t,S,E,I,R,D) and
        # abm_nets.py, which reads it positionally, is untouched.
        self.vaccinated_proportion_history = []
        # Instrumentation for the waning validation (fixed/stochastic modes):
        # [t, recovered_pool, recovered_waned, vaccinated_pool, vaccinated_waned].
        self.waning_events_history = []

        self.STAGE_UPDATE_VAR = 1
        self.INFINITY_TIME = self.config["simulation_metadata"]["INFINITY_TIME"]
        self.EXPOSED_TO_INFECTED_TIME = self.config["simulation_metadata"][
            "EXPOSED_TO_INFECTED_TIME"
        ]
        self.INFECTED_TO_RECOVERED_TIME = self.config["simulation_metadata"][
            "INFECTED_TO_RECOVERED_TIME"
        ]

        # --- Configurable immunity waning (R -> S and VACCINATED -> S) --------
        # Selected before the run via config.yaml; not learnable, not a
        # counterfactual. Natural immunity (I -> R -> S) and vaccine immunity
        # (S -> VACCINATED -> S) wane on independent durations/rates. Modes:
        #   "none"       -> no waning: S->E->I->R and S->VACCINATED stay put
        #   "fixed"      -> R -> S after RECOVERED_TO_SUSCEPTIBLE_TIME days,
        #                   VACCINATED -> S after VACCINATED_TO_SUSCEPTIBLE_TIME
        #   "stochastic" -> daily p = 1 - exp(-RECOVERED_WANING_RATE) for R,
        #                   daily p = 1 - exp(-VACCINATED_WANING_RATE) for VACC
        sim_md = self.config["simulation_metadata"]
        self.IMMUNITY_WANING_MODE = sim_md.get("IMMUNITY_WANING_MODE", "none")
        self.RECOVERED_TO_SUSCEPTIBLE_TIME = sim_md.get(
            "RECOVERED_TO_SUSCEPTIBLE_TIME", 100
        )
        self.VACCINATED_TO_SUSCEPTIBLE_TIME = sim_md.get(
            "VACCINATED_TO_SUSCEPTIBLE_TIME", 150
        )
        # RECOVERED_WANING_RATE supersedes the old WANING_RATE key; fall back to
        # it so waning_tests/set_waning_config.py keeps working unchanged.
        self.RECOVERED_WANING_RATE = sim_md.get(
            "RECOVERED_WANING_RATE", sim_md.get("WANING_RATE", 0.01)
        )
        self.VACCINATED_WANING_RATE = sim_md.get("VACCINATED_WANING_RATE", 0.00667)
        # kept for backward compatibility with any external reader
        self.WANING_RATE = self.RECOVERED_WANING_RATE
        # Sentinel "no scheduled transition" time, safely past the end of a run
        # (INFINITY_TIME in the config is smaller than num_steps_per_episode, so
        # it cannot be reused here for the fixed-mode timer).
        self.WANING_NO_TRANSITION_TIME = 10 * self.num_timesteps

        self.mode = self.config["simulation_metadata"]["EXECUTION_MODE"]
        self.st_bernoulli = StraightThroughBernoulli.apply

        self.calibration_mode = self.config['simulation_metadata']['calibration']

        # --- Counterfactual common random numbers (CRN) ----------------------
        # When GENERATING_COUNTERFACTUAL, every stochastic draw below is routed
        # through a torch.Generator seeded by (random_seed, _cf_iter, step,
        # call-site) -- NOT by COUNTERFACTUAL_TYPE. So for a given iteration two
        # CF policies that coincide at a step consume identical randomness, and
        # trajectory differences are attributable to the policy, not RNG drift.
        # _cf_iter is set per iteration by abm_nets.eval_net.
        self._crn_call_ids = {
            "init_infect": 11, "isolation": 23, "interv_school": 31,
            "interv_occ": 43, "vaccine": 57, "expose": 61, "waning": 79,
            "interv_occ_selective": 47, "vaccine_hh": 83,
        }

        # per-age-group daily NEW exposures [t, child_new, adult_new, elderly_new]
        # for the age-priority CF family. Diagnostic only.
        self.age_incidence_history = []
        # per-essentiality-group daily NEW exposures
        # [t, essential_new, nonessential_new, other_new] for CF 18-23.
        self.essential_incidence_history = []
        # int64 [N] essentiality code, lazily loaded by _require_agent_essential
        self._agent_essential = None
        # household-membership lookups, lazily loaded by _require_household_members
        # (only on a CF 25-28 path; never by calibration or CF 1-24).
        self._hh_members = None
        # cumulative dose-quota bookkeeping for the household-alloc family
        # (reset at t==0 inside apply_vaccines).
        self._cum_hh_quota = 0

    def _require_agent_essential(self):
        """int64 [N]: 0 = not in workforce, 1 = essential sector,
        2 = non-essential sector. Loaded once, on first use by a CF 18-23 path;
        never by calibration or CF 1-17. Fails loud. AGENT_OCC_CSV env (set by
        the Phase-2 essentiality run scripts) overrides the committed csv."""
        if self._agent_essential is not None:
            return self._agent_essential
        fips = self.config["simulation_metadata"]["POPULATION"]
        override = os.environ.get("AGENT_OCC_CSV", "").strip()
        if override:
            if not os.path.isfile(override):
                raise RuntimeError(f"AGENT_OCC_CSV={override!r} does not exist")
            hits = [override]
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
            hits = glob.glob(os.path.join(
                project_root, "data", "population_data", "*", f"{fips}_population.csv"))
        if len(hits) != 1:
            raise RuntimeError(
                f"_require_agent_essential: expected exactly one population csv "
                f"for {fips}, found {hits}")
        df = pd.read_csv(hits[0])
        n = int(self.config["simulation_metadata"]["num_agents"])
        if len(df) != n:
            raise RuntimeError(f"{hits[0]}: {len(df)} rows != num_agents {n}")
        occ = df["Occupations"]
        known = ESSENTIAL_SECTORS | NON_ESSENTIAL_SECTORS | {"Unemployed"}
        bad = set(occ.dropna().unique()) - known
        if bad:
            raise RuntimeError(f"{hits[0]}: unknown Occupations values {sorted(bad)}")
        code = np.zeros(n, dtype=np.int64)
        code[occ.isin(ESSENTIAL_SECTORS).to_numpy()] = 1
        code[occ.isin(NON_ESSENTIAL_SECTORS).to_numpy()] = 2
        self._agent_essential = torch.tensor(code, device=self.device)
        return self._agent_essential

    def _essential_group_masks(self):
        e = self._require_agent_essential()
        return (e == 1), (e == 2), (e == 0)   # essential, non-essential, other

    def _require_household_members(self):
        """(agent_to_hh [N] int64, hh_size_per_agent [N] int64, n_hh int) built
        once from the committed {fips}_population.csv 'Household ID' column (row
        order == agent index). Loaded lazily on the first CF 25-28 path; never by
        calibration or CF 1-24. Fails loud. Mirrors SpilloverHook.__init__
        (targeted_cf/spillover_hook.py; Tasks/household_protection.md sec 1.3)."""
        if self._hh_members is not None:
            return self._hh_members
        fips = self.config["simulation_metadata"]["POPULATION"]
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        hits = glob.glob(os.path.join(
            project_root, "data", "population_data", "*", f"{fips}_population.csv"))
        if len(hits) != 1:
            raise RuntimeError(
                f"_require_household_members: expected exactly one population csv "
                f"for {fips}, found {hits}")
        df = pd.read_csv(hits[0])
        n = int(self.config["simulation_metadata"]["num_agents"])
        if len(df) != n:
            raise RuntimeError(f"{hits[0]}: {len(df)} rows != num_agents {n}")
        if df["Household ID"].isna().any():
            raise RuntimeError(f"{hits[0]}: 'Household ID' column contains NaN")
        hh_id = df["Household ID"].to_numpy(dtype=np.int64)
        hh_size_col = df["Household Size"].to_numpy(dtype=np.int64)

        from collections import defaultdict
        groups = defaultdict(list)
        for agent_idx, hid in enumerate(hh_id):
            groups[hid].append(agent_idx)
        hh_ids_sorted = sorted(groups)
        seen = np.zeros(n, dtype=bool)
        agent_to_hh = np.full(n, -1, dtype=np.int64)
        hh_sizes = np.zeros(len(hh_ids_sorted), dtype=np.int64)
        for hh_idx, hid in enumerate(hh_ids_sorted):
            members = groups[hid]
            hh_sizes[hh_idx] = len(members)
            for m in members:
                if seen[m]:
                    raise RuntimeError(f"{hits[0]}: agent {m} appears in >1 household")
                seen[m] = True
                agent_to_hh[m] = hh_idx
        if not seen.all():
            raise RuntimeError(
                f"{hits[0]}: {int((~seen).sum())} agents not assigned to any household")
        decl = np.array([hh_size_col[groups[hid][0]] for hid in hh_ids_sorted], dtype=np.int64)
        bad = np.where(np.abs(hh_sizes - decl) > 1)[0]
        if bad.size:
            raise RuntimeError(
                f"{hits[0]}: {bad.size} households differ from declared 'Household Size' "
                f"by >1 (first: hh_idx {int(bad[0])}, actual {int(hh_sizes[bad[0]])}, "
                f"declared {int(decl[bad[0]])})")
        if hh_sizes.max() > 10:
            raise RuntimeError(f"{hits[0]}: household of size {int(hh_sizes.max())} > 10 (sanity)")

        agent_to_hh_t = torch.tensor(agent_to_hh, dtype=torch.long, device=self.device)
        hh_size_pa_t = torch.tensor(hh_sizes[agent_to_hh], dtype=torch.long, device=self.device)
        self._hh_members = (agent_to_hh_t, hh_size_pa_t, len(hh_ids_sorted))
        return self._hh_members

    def _age_group_masks(self, agents_ages):
        """(child, adult, elderly) boolean [N] masks over the 6-value age code
        (CHILD_CODES / ADULT_CODES / ELDERLY_CODES). Fails loud if any agent
        carries a code outside 0..5."""
        a = agents_ages.reshape(-1).long()
        child = torch.zeros_like(a, dtype=torch.bool)
        adult = torch.zeros_like(a, dtype=torch.bool)
        elderly = torch.zeros_like(a, dtype=torch.bool)
        for c in CHILD_CODES:
            child |= a == c
        for c in ADULT_CODES:
            adult |= a == c
        for c in ELDERLY_CODES:
            elderly |= a == c
        if not bool((child | adult | elderly).all()):
            bad = sorted(set(a.unique().tolist()) - set(VALID_AGE_CODES))
            raise RuntimeError(
                "_age_group_masks: age codes outside 0..5 for population "
                f"{self.config['simulation_metadata']['POPULATION']}: {bad}"
            )
        return child, adult, elderly

    def _generating_cf(self):
        return str(self.config["simulation_metadata"].get(
            "GENERATING_COUNTERFACTUAL", False)).lower() in ("true", "1")

    def _crn_gen(self, t, call_id, device=None):
        """Deterministic torch.Generator for one CF stochastic call site,
        seeded by (random_seed, _cf_iter, t, call_id) -- independent of
        COUNTERFACTUAL_TYPE (common random numbers across CF policies)."""
        md = self.config["simulation_metadata"]
        base = int(md.get("random_seed", md.get("SEED", 42)))
        it = int(md.get("_cf_iter", 0))
        cid = self._crn_call_ids.get(call_id, 97)
        s = (((base & 0xFFFFF) * 1_000_003 + it) * 1_000_003 + int(t)) * 131 + cid
        g = torch.Generator(device=device or self.device)
        g.manual_seed(int(s & 0x7FFF_FFFF_FFFF_FFFF))
        return g

    def _lam(
        self,
        x_i,
        x_j,
        edge_attr,
        t,
        R,
        SFSusceptibility,
        SFInfector,
        lam_gamma_integrals,
    ):
        S_A_s = SFSusceptibility[x_i[:, 0].long()]
        A_s_i = SFInfector[x_j[:, 1].long()]
        B_n = edge_attr[1, :]
        integrals = torch.zeros_like(B_n)
        infected_idx = x_j[:, 2].bool()
        infected_times = t - x_j[infected_idx, 3] - 1
        infected_times = infected_times.clamp(min=0, max=lam_gamma_integrals.size(0) - 1)

        integrals[infected_idx] = lam_gamma_integrals[infected_times.long()]
        I_bar = x_i[:, 4].view(-1)
        I_bar = torch.clamp(I_bar, min=1e-5)

        will_isolate = x_i[:, 6]
        not_isolated = 1 - will_isolate

        if self.mode == "llm":
            res = (
                R * S_A_s * A_s_i * B_n * integrals / I_bar
            )
        else:
            res = R * S_A_s * A_s_i * B_n * integrals / I_bar

        return res.view(-1, 1)

    def message(
        self,
        x_i,
        x_j,
        edge_attr,
        t,
        R,
        SFSusceptibility,
        SFInfector,
        lam_gamma_integrals,
    ):
        return self._lam(
            x_i, x_j, edge_attr, t, R, SFSusceptibility, SFInfector, lam_gamma_integrals
        )

    def update_stages(self, t, current_stages, agents_next_stage_times, newly_exposed_today):
        transition_to_infected = self.INFECTED_VAR*(agents_next_stage_times <= t) + self.EXPOSED_VAR*(agents_next_stage_times > t)
        transition_to_mortality_or_recovered = self.RECOVERED_VAR*(agents_next_stage_times <= t) + self.INFECTED_VAR*(agents_next_stage_times > t)

        stage_progression = (current_stages == self.SUSCEPTIBLE_VAR)*self.SUSCEPTIBLE_VAR \
            + (current_stages == self.RECOVERED_VAR)*self.RECOVERED_VAR + (current_stages == self.MORTALITY_VAR)*self.MORTALITY_VAR \
            + (current_stages == self.VACCINATED_VAR)*self.VACCINATED_VAR \
            + (current_stages == self.EXPOSED_VAR)*transition_to_infected \
            + (current_stages == self.INFECTED_VAR)*transition_to_mortality_or_recovered

        current_stages = newly_exposed_today*self.EXPOSED_VAR + stage_progression
        return current_stages

    def update_transition_times(self, t, agents_next_stage_times, newly_exposed_today, current_stages):
        """Note: not differentiable"""
        exposed_to_infected_time = self.EXPOSED_TO_INFECTED_TIME
        new_transition_times = torch.clone(agents_next_stage_times) 
        curr_stages = torch.clone(current_stages).long()
        new_transition_times[(curr_stages==self.INFECTED_VAR)*(agents_next_stage_times == t)] = self.INFINITY_TIME 
        new_transition_times[(curr_stages==self.EXPOSED_VAR)*(agents_next_stage_times == t)] = t+self.INFECTED_TO_RECOVERED_TIME 
        return newly_exposed_today*(t+1+exposed_to_infected_time) + (1 - newly_exposed_today)*new_transition_times

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        timestep_tensor = torch.tensor([timestep])
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)
        one_hot_tensor = one_hot_tensor.view(1, -1)
        return one_hot_tensor.to(self.device)[0]

    def update_infected_times(self, t, agents_infected_times, newly_exposed_today):
        """Note: not differentiable"""
        updated_infected_times = torch.clone(agents_infected_times).to(
            agents_infected_times.device
        )

        updated_infected_times[newly_exposed_today.bool()] = t

        return updated_infected_times

    def get_stage_proportions(self, t, current_stages):
        total_people = current_stages.shape[0]
        counts = torch.stack([
            (current_stages == self.SUSCEPTIBLE_VAR).sum(),
            (current_stages == self.EXPOSED_VAR).sum(),
            (current_stages == 2).sum(),
            (current_stages == self.RECOVERED_VAR).sum(),
            (current_stages == 4).sum(),
        ])
        proportions = counts.float() / total_people

        # proportion_history stays 5-wide (t,S,E,I,R,D) for abm_nets.py. Once
        # vaccination starts, S+E+I+R+D no longer sums to 1 - the remainder is
        # the vaccinated fraction, tracked here in its own list.
        self.proportion_history.append([t] + proportions.detach().cpu().tolist())
        vaccinated_fraction = (
            (current_stages == self.VACCINATED_VAR).sum().float() / total_people
        )
        self.vaccinated_proportion_history.append(
            [t, vaccinated_fraction.detach().cpu().item()]
        )

    def _merge_vaccinated_column(self, df):
        """Left-join the vaccinated fraction onto a proportions dataframe."""
        if self.vaccinated_proportion_history:
            vdf = pd.DataFrame(
                self.vaccinated_proportion_history, columns=["t", "vaccinated"]
            )
            df = df.merge(vdf, on="t", how="left")
        return df

    def save_proportions_to_disk(self, output_path):
        if self.proportion_history:
            df = pd.DataFrame(self.proportion_history,
                            columns=["t", "susceptible", "exposed", "infected", "recovered", "dead"])
            df = self._merge_vaccinated_column(df)
            df.to_csv(output_path, index=False)

    def save_waning_events_to_disk(self, output_path):
        """Per-step waning instrumentation (fixed/stochastic modes only)."""
        if self.waning_events_history:
            pd.DataFrame(
                self.waning_events_history,
                columns=["t", "recovered_pool", "recovered_waned",
                         "vaccinated_pool", "vaccinated_waned"],
            ).to_csv(output_path, index=False)

    def save_proportions_to_disk2(self, output_path, epoch_num):
        if self.proportion_history:
            cols = [
                "t",
                f"susceptible_{epoch_num}",
                f"exposed_{epoch_num}",
                f"infected_{epoch_num}",
                f"recovered_{epoch_num}",
                f"dead_{epoch_num}"
            ]

            new_df = pd.DataFrame(self.proportion_history, columns=cols)
            if self.vaccinated_proportion_history:
                vdf = pd.DataFrame(
                    self.vaccinated_proportion_history,
                    columns=["t", f"vaccinated_{epoch_num}"],
                )
                new_df = new_df.merge(vdf, on="t", how="left")

            if os.path.exists(output_path):
                existing_df = pd.read_csv(output_path)

                if "t" in existing_df.columns:
                    final_df = pd.merge(existing_df, new_df, on="t", how="outer")
                else:
                    final_df = pd.concat([existing_df, new_df.drop(columns=["t"])], axis=1)
            else:
                final_df = new_df

            final_df.to_csv(output_path, index=False)

    def save_age_incidence_to_disk(self, output_path, epoch_num):
        """Per-iteration wide dump of per-age-group daily NEW exposures
        (age-priority CF family). Mirrors save_proportions_to_disk2."""
        if not self.age_incidence_history:
            raise RuntimeError(
                f"save_age_incidence_to_disk: empty age_incidence_history "
                f"(path={output_path}, iter={epoch_num})"
            )
        cols = ["t", f"child_new_{epoch_num}",
                f"adult_new_{epoch_num}", f"elderly_new_{epoch_num}"]
        new_df = pd.DataFrame(self.age_incidence_history, columns=cols)
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            final_df = pd.merge(existing_df, new_df, on="t", how="outer")
        else:
            final_df = new_df
        final_df.to_csv(output_path, index=False)

    def save_essential_incidence_to_disk(self, output_path, epoch_num):
        """Per-iteration wide dump of per-essentiality-group daily NEW exposures
        (work-essentiality CF family). Mirrors save_age_incidence_to_disk."""
        if not self.essential_incidence_history:
            raise RuntimeError(
                f"save_essential_incidence_to_disk: empty essential_incidence_history "
                f"(path={output_path}, iter={epoch_num})"
            )
        cols = ["t", f"essential_new_{epoch_num}",
                f"nonessential_new_{epoch_num}", f"other_new_{epoch_num}"]
        new_df = pd.DataFrame(self.essential_incidence_history, columns=cols)
        if os.path.exists(output_path):
            existing_df = pd.read_csv(output_path)
            final_df = pd.merge(existing_df, new_df, on="t", how="outer")
        else:
            final_df = new_df
        final_df.to_csv(output_path, index=False)

    def modify_initial_infected(self, current_stages, proportion, tau=0.5, hard=True):
        device = current_stages.device
        N = int(self.config['simulation_metadata']['num_agents'])

        new_stages = current_stages.clone()
        new_stages[:, 0] = 0.0

        logits = torch.zeros(N, 2, device=device)
        val = proportion / (1 - proportion)
        if isinstance(val, torch.Tensor):
            logits[:, 1] = torch.log(val.to(device))
        else:
            logits[:, 1] = torch.log(torch.tensor(val, device=device, dtype=torch.float))

        if self._generating_cf():
            # CRN gumbel-softmax so initial infections match across CF policies.
            u = torch.rand(logits.shape, generator=self._crn_gen(0, "init_infect"),
                           device=logits.device, dtype=logits.dtype)
            gum = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            soft = F.softmax((logits + gum) / tau, dim=-1)
            if hard:
                oh = torch.zeros_like(soft).scatter_(
                    -1, soft.argmax(dim=-1, keepdim=True), 1.0)
                samples = (oh - soft).detach() + soft
            else:
                samples = soft
        else:
            samples = F.gumbel_softmax(logits, tau=tau, hard=hard)
        infected_mask = samples[:, 1]

        new_stages[:, 0] = 2.0 * infected_mask

        return new_stages

    def get_mean_agent_interactions(self, agents_ages):
        # NOTE: this split is mis-labelled w.r.t. the real age encoding
        # (child == code<1 == 20t29; elderly == code>4 == U19). It is LEFT AS-IS
        # on purpose: the resulting per-agent interaction constant feeds I_bar in
        # _lam and is therefore baked into every committed calibration. Fixing it
        # here would change the factual dynamics and invalidate reused
        # calibrated params. The corrected child/adult/elderly split lives in
        # _age_group_masks and is used only by the diagnostic outputs.
        ADULT_LOWER_INDEX, ADULT_UPPER_INDEX = (
            1,
            4,
        )

        agents_mean_interactions = 0 * torch.ones(size=agents_ages.shape)
        mean_int_ran_mu = torch.tensor([2, 3, 4]).float()

        child_agents = (agents_ages < ADULT_LOWER_INDEX).view(-1)
        adult_agents = torch.logical_and(
            agents_ages >= ADULT_LOWER_INDEX, agents_ages <= ADULT_UPPER_INDEX
        ).view(-1)
        elderly_agents = (agents_ages > ADULT_UPPER_INDEX).view(-1)

        agents_mean_interactions[child_agents.bool(), 0] = mean_int_ran_mu[0]
        agents_mean_interactions[adult_agents.bool(), 0] = mean_int_ran_mu[1]
        agents_mean_interactions[elderly_agents.bool(), 0] = mean_int_ran_mu[2]

        return agents_mean_interactions

    def update_initial_times(self, agents_next_stage_times, agents_infected_time, agents_stages):
        infected_to_recovered_time = self.INFECTED_TO_RECOVERED_TIME
        exposed_to_infected_time = self.EXPOSED_TO_INFECTED_TIME

        agents_infected_time[agents_stages==self.EXPOSED_VAR] = -1
        agents_infected_time[agents_stages==self.INFECTED_VAR] = -1*self.EXPOSED_TO_INFECTED_TIME
        agents_next_stage_times[agents_stages==self.EXPOSED_VAR] = exposed_to_infected_time
        agents_next_stage_times[agents_stages==self.INFECTED_VAR] = infected_to_recovered_time

        return agents_infected_time, agents_next_stage_times

    def soft_eq(self, x, target, temperature=0.1):
        diff = torch.abs(x - target)
        return torch.exp(-diff / temperature)

    def combined(self, x, target, t, agents_next_stage_times):
        soft = self.soft_eq(x, target)
        hard = (x == target).float() * (agents_next_stage_times <= t).float()

        return hard.detach() + soft - soft.detach()

    def update_number_of_dead(self, daily_dead, current_stages, agents_next_stage_times, t, newly_exposed_today):
        if self.calibration_mode:
            mortality_rate = self.calibrate_M.to(self.device)
        else:
            mortality_rate = self.learnable_args["M"]

        mask = self.combined(current_stages, self.INFECTED_VAR, t, agents_next_stage_times)

        new_death_recovered_today = (current_stages * mask) / self.INFECTED_VAR


        NEW_DEATHS_TODAY = mortality_rate * new_death_recovered_today.sum()

        daily_dead = (
            daily_dead
            + self._generate_one_hot_tensor(t, self.num_timesteps) * NEW_DEATHS_TODAY.squeeze()
        )

        return daily_dead

    def update_adjacency_matrix(self, state, combined_net):
        source_nodes = combined_net[:, 0]
        target_nodes = combined_net[:, 1]

        edge_list = torch.stack((source_nodes, target_nodes), dim=0).to(self.device)

        edge_attr = torch.ones(2, edge_list.size(1)).to(self.device) 

        adjacency_matrix_path = ["network", "agent_agent", "infection_network", "adjacency_matrix"]
        adjacency_matrix = (edge_list, edge_attr)

        return set_by_path(state, adjacency_matrix_path, adjacency_matrix)

    def _load_single_net(self, path):
        df = pd.read_pickle(path)
        if hasattr(df, 'edges'):
            df = pd.DataFrame(df.edges(), columns=["node1", "node2"])
        return torch.tensor(df.values, device=self.device, dtype=torch.long)

    def _preload_all_networks(self):
        """Load all time-step networks into a dictionary of tensors on the GPU."""
        nets = {'occ': [], 'rand': []}
        population = self.config['simulation_metadata']['POPULATION']
        for t in range(self.num_timesteps):
            nets['occ'].append(self._load_single_net(f"{self.networks_dir}/{population}/mobility_networks/occnets/{t}.pkl"))
            nets['rand'].append(self._load_single_net(f"{self.networks_dir}/{population}/mobility_networks/randnets/{t}.pkl"))
        return nets

    def apply_intervention_fast(self, intervention, edges, t=0, net_id="net",
                                essential_edge_mask=None):
        if edges.size(0) == 0: return edges

        # CRN in counterfactual mode: the closure fraction and the kept-edge
        # subset for (iteration, step, this network) are identical across CF
        # policies -- so two policies applying the *same* intervention value at
        # step t get the *same* realised network.
        cf = self._generating_cf()
        g_cpu = self._crn_gen(t, f"interv_{net_id}", device="cpu") if cf else None
        g_dev = self._crn_gen(t, f"interv_{net_id}") if cf else None

        # Edge-removal fraction `rm` by policy state:
        #   intervention == 0 (open)   -> rm ~ N(OPEN_RM_MEAN,   RM_SD)  truncated to [0.00, 0.25]
        #   intervention == 1 (closed) -> rm ~ N(CLOSED_RM_MEAN, RM_SD)  truncated to [0.75, 1.00]
        # A SINGLE standardized shock `eps` is drawn from the CRN generator and
        # reused for both branches, so at a given (cf_iter, step, network):
        #   * within one CF's 30 iterations the severity varies (Monte Carlo), but
        #   * every CF policy that is "open" there realises the SAME open fraction,
        #     and every policy "closed" there the SAME closed fraction
        #     (the CRN generator key excludes COUNTERFACTUAL_TYPE).
        # Was: both branches drew N(0.5, 0.15) then clamped into the band, which
        # put ~95% of the mass exactly on the inner edge (rm == 0.25 for open,
        # rm == 0.75 for closed) -- closure severity was effectively deterministic
        # and the open/closed contrast the weakest the band allows.
        # See results/LossCompare_run/INTERVENTION_STOCHASTICITY.md.
        OPEN_RM_MEAN, CLOSED_RM_MEAN, RM_SD = 0.125, 0.875, 0.0625
        eps = torch.randn((1,), generator=g_cpu)

        # Work-essentiality CF 19/20/22/23: selective closure. NON-ESSENTIAL
        # workplace edges always take the CLOSED removal; ESSENTIAL workplace
        # edges follow `intervention` (the factual occ value for that day) -- so
        # essential workers keep exactly the occ contact they had in the factual
        # while non-essential workplaces shut. Dedicated CRN generator so the two
        # subsets are reproducible across the selective CF types.
        if essential_edge_mask is not None:
            open_rm = torch.clamp(OPEN_RM_MEAN + RM_SD * eps, 0.0, 0.25)
            closed_rm = torch.clamp(CLOSED_RM_MEAN + RM_SD * eps, 0.75, 1.0)
            ess_rm = open_rm if intervention == 0 else closed_rm
            e_idx = essential_edge_mask.nonzero(as_tuple=True)[0]
            ne_idx = (~essential_edge_mask).nonzero(as_tuple=True)[0]
            g_sel = self._crn_gen(t, "interv_occ_selective") if cf else None
            ke = int(e_idx.numel() * (1.0 - ess_rm.item()))
            kne = int(ne_idx.numel() * (1.0 - closed_rm.item()))
            kept_e = e_idx[torch.randperm(e_idx.numel(), device=self.device, generator=g_sel)[:ke]]
            kept_ne = ne_idx[torch.randperm(ne_idx.numel(), device=self.device, generator=g_sel)[:kne]]
            return edges[torch.cat([kept_e, kept_ne])]

        if intervention == 0:
            sample = torch.clamp(OPEN_RM_MEAN + RM_SD * eps, 0.0, 0.25).to(self.device)
        else:
            sample = torch.clamp(CLOSED_RM_MEAN + RM_SD * eps, 0.75, 1.0).to(self.device)

        keep_frac = (1.0 - sample).item()

        k = int(edges.size(0) * keep_frac)
        idx = torch.randperm(edges.size(0), device=self.device, generator=g_dev)[:k]
        return edges[idx]


    def apply_vaccines(self, current_stages, num_vaccines, tau=0.1, t=0,
                       agents_ages=None, priority=None, essential_first=False,
                       household_order=None):
        device = current_stages.device
        N = current_stages.shape[0]
        if sum(x is not None and x is not False
               for x in (priority, essential_first or None, household_order)) > 1:
            raise RuntimeError(
                "apply_vaccines: priority / essential_first / household_order are "
                "mutually exclusive")

        is_susceptible = self.soft_eq(current_stages, self.SUSCEPTIBLE_VAR, temperature=tau)
        logits = torch.log(is_susceptible + 1e-10).view(N, 1)

        if self._generating_cf():
            u = torch.rand(logits.shape, generator=self._crn_gen(t, "vaccine"),
                           device=logits.device, dtype=logits.dtype)
        else:
            u = torch.rand_like(logits)
        gumbels = -torch.log(-torch.log(u + 1e-10) + 1e-10)
        y = logits + gumbels

        # Age-priority CF family (CF 12-17): add a large rank offset to the
        # candidate score of SUSCEPTIBLE agents only, so topk drains the
        # highest-rank age band before spilling to the next. Ties within a band
        # are still broken by the (CRN) Gumbel noise. priority=None -> the
        # committed uniform-allocation behaviour, byte-for-byte.
        if priority is not None:
            if agents_ages is None:
                raise RuntimeError("apply_vaccines: priority set but agents_ages is None")
            a = agents_ages.reshape(-1).long()
            unknown = sorted(set(a.unique().tolist()) - set(VALID_AGE_CODES))
            if unknown:
                raise RuntimeError(
                    f"apply_vaccines: unknown age codes {unknown} for population "
                    f"{self.config['simulation_metadata']['POPULATION']}"
                )
            susc = (current_stages.reshape(-1) == self.SUSCEPTIBLE_VAR).float()
            rank = torch.zeros(N, device=device)
            for code, r in priority.items():
                rank = rank + float(r) * (a == code).float()
            BIG = 1.0e4   # >> |logits| (~23) and Gumbel noise (~O(1))
            y = (y.reshape(-1) + BIG * rank * susc).view(N, 1)

        # Work-essentiality CF 18/20/21/23: essential-sector workers (code 1)
        # vaccinated before anyone else. Same susceptibles-only offset trick.
        if essential_first:
            e = self._require_agent_essential()
            susc = (current_stages.reshape(-1) == self.SUSCEPTIBLE_VAR).float()
            rank = (e == 1).float()
            y = (y.reshape(-1) + 1.0e4 * rank * susc).view(N, 1)

        # Household-level allocation CF 25-28: a per-household key -- broadcast to
        # every member -- added to the SUSCEPTIBLE score, so topk drains whole
        # households in a fixed campaign order (cohabitants always consecutive).
        # The key is drawn ONCE per iteration (t=0), CRN-keyed, and is independent
        # of COUNTERFACTUAL_TYPE; ties within a household stay on the CF7-shared
        # Gumbel draw. household_order=None -> unchanged committed behaviour.
        if household_order is not None:
            agent_to_hh, hh_size_pa, n_hh = self._require_household_members()
            susc = (current_stages.reshape(-1) == self.SUSCEPTIBLE_VAR).float()
            g_hh = self._crn_gen(0, "vaccine_hh")
            hh_tiebreak = torch.rand(n_hh, generator=g_hh, device=device)      # (0,1)

            if household_order == "random":
                hh_key = hh_tiebreak
            elif household_order == "elderly_home_first":
                if agents_ages is None:
                    raise RuntimeError("apply_vaccines: elderly_home_first needs agents_ages")
                a = agents_ages.reshape(-1).long()
                elderly_agent = torch.zeros(N, device=device)
                for c in ELDERLY_CODES:
                    elderly_agent = elderly_agent + (a == c).float()
                hh_has = torch.zeros(n_hh, device=device)
                hh_has.scatter_add_(0, agent_to_hh, elderly_agent)
                hh_key = 2.0 * (hh_has > 0).float() + hh_tiebreak
            elif household_order == "largest_first":
                hh_size = torch.zeros(n_hh, device=device)
                hh_size.scatter_add_(0, agent_to_hh, torch.ones(N, device=device))
                hh_key = hh_size + 0.1 * hh_tiebreak
            elif household_order == "essential_home_first":
                e = self._require_agent_essential()
                hh_has = torch.zeros(n_hh, device=device)
                hh_has.scatter_add_(0, agent_to_hh, (e == 1).float())
                hh_key = 2.0 * (hh_has > 0).float() + hh_tiebreak
            else:
                raise RuntimeError(
                    f"apply_vaccines: unknown household_order {household_order!r}")

            per_agent_hh_key = hh_key[agent_to_hh]
            BIG = 1.0e4   # >> |logits| (~23) and Gumbel noise (~O(1))
            y = (y.reshape(-1) + BIG * per_agent_hh_key * susc).view(N, 1)

        num_vax_int = int(num_vaccines.item()) if torch.is_tensor(num_vaccines) else int(num_vaccines)
        n_susc_now = int(is_susceptible.sum().item())
        if household_order is not None:
            # fail loud if the (pre-clamp) cumulative daily dose quota over the
            # whole run exceeds the number of agents -- the dose series or the
            # population is wrong (not a silent clamp).
            if t == 0:
                self._cum_hh_quota = 0
            self._cum_hh_quota += num_vax_int
            if t == self.num_timesteps - 1 and self._cum_hh_quota > N:
                raise RuntimeError(
                    f"apply_vaccines: household-alloc cumulative dose quota "
                    f"{self._cum_hh_quota} > num_agents {N} for population "
                    f"{self.config['simulation_metadata']['POPULATION']} -- "
                    f"dose series / population mismatch")
        num_vax_int = min(num_vax_int, n_susc_now)
        
        if num_vax_int <= 0:
            return current_stages

        _, indices = torch.topk(y.view(-1), num_vax_int)
        
        hard_mask = torch.zeros(N, 1, device=device)
        hard_mask[indices] = 1.0
        
        soft_probs = torch.sigmoid(logits)
        vax_mask = (hard_mask - soft_probs).detach() + soft_probs

        # Vaccination now moves S -> VACCINATED (a distinct stage), not S -> R,
        # so vaccine-derived immunity can wane on its own schedule. Selection,
        # counts and the straight-through mask are unchanged.
        stage_delta = (self.VACCINATED_VAR - self.SUSCEPTIBLE_VAR) * vax_mask
        updated_stages = current_stages + stage_delta

        return updated_stages

    def get_age_stage_proportions(self, t, current_stages, agents_ages):
            # Corrected child / adult / elderly split (was mis-wired: child ==
            # code<1 == 20t29, elderly == code>4 == U19). Diagnostic output only
            # -> safe to fix without touching the calibrated dynamics.
            child_agents, adult_agents, elderly_agents = self._age_group_masks(agents_ages)

            masks = [child_agents, adult_agents, elderly_agents]
            row = [t]

            for mask in masks:
                total_people_in_group = mask.sum()
                if total_people_in_group > 0:
                    counts = torch.stack([
                        (current_stages[mask] == self.SUSCEPTIBLE_VAR).sum(),
                        (current_stages[mask] == self.EXPOSED_VAR).sum(),
                        (current_stages[mask] == 2).sum(),
                        (current_stages[mask] == self.RECOVERED_VAR).sum(),
                        (current_stages[mask] == 4).sum(),
                    ])
                    proportions = counts.float() / total_people_in_group
                    row.extend(proportions.detach().cpu().tolist())
                else:
                    row.extend([0.0] * 5)

            self.age_proportion_history.append(row)

    def forward(self, state, action=None):
        input_variables = self.input_variables
        t = int(state["current_step"])

        generating_counterfactual = self.config['simulation_metadata']['GENERATING_COUNTERFACTUAL']
        cf_type = self.config['simulation_metadata']['COUNTERFACTUAL_TYPE']
        with_k = self.config['simulation_metadata']['WITH_K']
        with_vacc = self.config['simulation_metadata']['WITH_VACC']
        
        intervention_df = pd.read_csv(f"{self.config['simulation_metadata']['population_dir']}/intervention.csv")
        curr_intervention = intervention_df[intervention_df['t'] == t]

        if len(curr_intervention) == 0:
            raise RuntimeError(
                f"intervention.csv has no row for t={t} "
                f"({self.config['simulation_metadata']['population_dir']})"
            )
        school_intervention = curr_intervention.iloc[0]['school_intervention']
        occ_intervention = curr_intervention.iloc[0]['occ_intervention']
        num_vaccines = curr_intervention.iloc[0]['vaccines']

        # Age-priority CF family (12-17): which AGE_PRIORITY ordering (if any)
        # applies to apply_vaccines this step. Uniform for every other mode.
        vacc_priority = None
        # Work-essentiality CF family (18-23): essential-first vaccine allocation
        # and/or non-essential-only workplace closure.
        essential_first = False
        essential_edge_mask = None
        # Household-level allocation CF family (25-28): whole-household dose
        # selection order (None -> individual, the committed behaviour).
        household_order = None

        if generating_counterfactual:
            if (t == 0):
                print(f"Counterfactual Type: {cf_type}")

            logic_map = {
                1: (0, 0),
                2: (1, 0),
                3: (0, 1),
                4: (1, 1),
                5: (0, "F"),
                6: (0, "CF"),
                7: ("F", "F"),
                8: ("CF", "F"),
                9: ("F", "CF"),
                10: ("CF", "CF"),
                11: ("F", "F"),
                # age-priority vaccination x school/work NPI
                12: ("F", "F"),   # elderly-first,     school factual, work factual
                13: ("F", "F"),   # working-age-first, school factual, work factual
                14: (1, 0),       # working-age-first, school closed,  work open
                15: (0, 0),       # working-age-first, school open,    work open
                16: (0, 1),       # elderly-first,     school open,    work closed
                17: (1, 1),       # elderly-first,     school closed,  work closed
                # work-essentiality: vaccine allocation x targeted-vs-full occ closure.
                # For CF_SELECTIVE_OCC types the occ lever stays FACTUAL; the
                # selective branch in apply_intervention_fast shuts non-essential
                # workplaces and leaves essential workplaces at their factual
                # severity. CF 24 forces ALL workplaces closed (not selective).
                18: ("F", "F"),   # essential-first vacc; occ factual;              school factual
                19: ("F", "F"),   # uniform vacc;         non-ess occ closed (sel); school factual
                20: ("F", "F"),   # essential-first;      non-ess occ closed (sel); school factual
                21: (0, 0),       # essential-first;      occ open;                 school open
                22: (1, "F"),     # uniform;              non-ess occ closed (sel); school closed
                23: (1, "F"),     # essential-first;      non-ess occ closed (sel); school closed
                24: ("F", 1),     # uniform;              ALL occ closed;           school factual  (retained-benefit ref)
                # household-level allocation: whole-household dose bolus x NPI factual.
                25: ("F", "F"),   # random household order
                26: ("F", "F"),   # elderly-home first
                27: ("F", "F"),   # largest households first
                28: ("F", "F"),   # essential-worker homes first
            }

            if cf_type in logic_map:
                s_logic, o_logic = logic_map[cf_type]

                if s_logic == "CF":
                    school_intervention = 1 - school_intervention
                elif s_logic != "F":
                    school_intervention = s_logic

                if o_logic == "CF":
                    occ_intervention = 1 - occ_intervention
                elif o_logic != "F":
                    occ_intervention = o_logic

            if cf_type == 11:
                num_vaccines = 0

            if cf_type in CF_VACC_PRIORITY:
                vacc_priority = AGE_PRIORITY[CF_VACC_PRIORITY[cf_type]]

            if cf_type in CF_ESSENTIAL_VACC:
                essential_first = True
            if cf_type in CF_SELECTIVE_OCC:
                e = self._require_agent_essential()
                occ_t = self.networks['occ'][t]
                em_u = e[occ_t[:, 0]] == 1
                em_v = e[occ_t[:, 1]] == 1
                # within-sector-edge invariant: both endpoints agree unless one
                # endpoint is a non-worker (code 0) that slipped into an occ net
                if not bool(((e[occ_t[:, 0]] == e[occ_t[:, 1]]) |
                             (e[occ_t[:, 0]] == 0) | (e[occ_t[:, 1]] == 0)).all()):
                    raise RuntimeError(
                        f"occ net t={t} has an edge crossing the essential boundary "
                        f"({self.config['simulation_metadata']['POPULATION']})")
                essential_edge_mask = em_u & em_v

            if cf_type in HH_ALLOC_CF_TYPES:
                household_order = CF_HH_ORDER[cf_type]

        school_net = self.apply_intervention_fast(school_intervention, self.school_net, t, "school")
        occ_net = self.apply_intervention_fast(occ_intervention, self.networks['occ'][t], t, "occ",
                                               essential_edge_mask=essential_edge_mask)

        combined_net = torch.cat([
                    school_net, 
                    occ_net, 
                    self.networks['rand'][t], 
                    self.household_net
                ], dim=0)

        state = self.update_adjacency_matrix(state, combined_net)

        time_step_one_hot = self._generate_one_hot_tensor(t, self.num_timesteps)

        week_id = int(t / 7)
        week_one_hot = self._generate_one_hot_tensor(week_id, self.num_weeks)

        if self.calibration_mode:
            R_tensor = self.calibrate_R2.to(self.device)
            initial_infection_rate = self.calibrate_infected_proportion.to(self.device)
            k = self.calibrate_k.to(self.device)
        else:
            R_tensor = self.learnable_args["R2"]
            initial_infection_rate = self.learnable_args["infected_proportion"]
            k = self.learnable_args["k"]
        
        if (t == 0):
            self.proportion_history = []
            self.age_proportion_history = []
            self.vaccinated_proportion_history = []
            self.waning_events_history = []
            self.age_incidence_history = []
            self.essential_incidence_history = []

        R = (R_tensor.T * week_one_hot).sum()

        current_stages = state['agents']['citizens']['disease_stage']
        agents_ages = get_by_path(state, re.split("/", input_variables["age"]))
        agents_next_stage_times = state['agents']['citizens']['next_stage_time']
        agents_infected_time = state['agents']['citizens']['infected_time']

        daily_deaths = get_by_path(
            state, re.split("/", input_variables["daily_deaths"])
        )

        if (t == 0):
            current_stages = self.modify_initial_infected(current_stages, initial_infection_rate)
            agents_infected_time, agents_next_stage_times = self.update_initial_times(agents_next_stage_times, agents_infected_time, current_stages)
            
            # Load residence commuter mapping for Phase 1 and 2
            self.commuters = []
            population = self.config['simulation_metadata']['POPULATION']
            commute_file = f"{self.networks_dir}/{population}/mobility_networks/occnets/{population}_residence_commute_data.txt"
            if os.path.exists(commute_file):
                try:
                    with open(commute_file, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                parts = line.split(',')
                                if len(parts) == 2:
                                    agent_id = int(parts[0])
                                    work_county = parts[1].strip()
                                    if work_county != population:
                                        self.commuters.append((agent_id, work_county))
                except Exception as e:
                    print(f"Warning: Failed to load commuter mapping: {e}")

        SFSusceptibility = get_by_path(
            state, re.split("/", input_variables["SFSusceptibility"])
        )
        SFInfector = get_by_path(state, re.split("/", input_variables["SFInfector"]))
        all_lam_gamma = get_by_path(
            state, re.split("/", input_variables["lam_gamma_integrals"])
        )

        agents_mean_interactions_split = self.get_mean_agent_interactions(agents_ages)

        all_edgelist, all_edgeattr = get_by_path(
            state, re.split("/", input_variables["adjacency_matrix"])
        )

        daily_infected = get_by_path(
            state, re.split("/", input_variables["daily_infected"])
        )

        agents_infected_index = torch.logical_and(
            current_stages > self.SUSCEPTIBLE_VAR, current_stages < self.RECOVERED_VAR
        )

        will_isolate = action["citizens"]["isolation_decision"]

        all_node_attr = (
            torch.stack(
                (
                    agents_ages.to(self.device),
                    current_stages.detach(),
                    agents_infected_index,
                    agents_infected_time,
                    agents_mean_interactions_split.to(self.device),
                    torch.unsqueeze(
                        torch.arange(self.config["simulation_metadata"]["num_agents"]),
                        1,
                    ).to(
                        self.device
                    ),
                    will_isolate,
                )
            )
            .transpose(0, 1)
            .squeeze()
        )

        agents_data = Data(
            all_node_attr, edge_index=all_edgelist, edge_attr=all_edgeattr, t=t
        )

        new_transmission = self.propagate(
            agents_data.edge_index,
            x=agents_data.x,
            edge_attr=agents_data.edge_attr,
            t=agents_data.t,
            R=R,
            SFSusceptibility=SFSusceptibility,
            SFInfector=SFInfector,
            lam_gamma_integrals=all_lam_gamma.squeeze(),
        )

        prob_not_infected = torch.exp(-1 * new_transmission)
        probs = torch.hstack((1 - prob_not_infected, prob_not_infected))

        if self._generating_cf():
            # CRN exposure draw (fixed size N, no gradient needed in CF mode).
            potentially_exposed_today = torch.bernoulli(
                probs, generator=self._crn_gen(t, "expose")
            )[:, 0].to(self.device)
        else:
            potentially_exposed_today = self.st_bernoulli(probs)[:, 0].to(
                self.device
            )

        newly_exposed_today = (
            current_stages == self.SUSCEPTIBLE_VAR
        ).squeeze() * potentially_exposed_today

        metro_phase = self.config['simulation_metadata'].get('metro_calibration_phase', 0)
        population = self.config['simulation_metadata']['POPULATION']
        extra_infections = 0.0
        num_to_expose = 0

        if metro_phase == 2:
            state_prefix = population[:2]
            csv_path = f"results/{state_prefix}/{t}.csv"
            num_incoming_infected = 0.0
            if os.path.exists(csv_path):
                try:
                    df_t = pd.read_csv(csv_path)
                    row = df_t[df_t['destination_county'].astype(str).str.zfill(5) == population]
                    if not row.empty:
                        num_incoming_infected = float(row.iloc[0]['num_infected'])
                except Exception as e:
                    pass
            
            P = potentially_exposed_today.mean()
            extra_infections = num_incoming_infected * P
            
            if extra_infections > 0:
                susceptible_indices = (current_stages.squeeze() == self.SUSCEPTIBLE_VAR).nonzero(as_tuple=True)[0]
                if len(susceptible_indices) > 0:
                    num_to_expose = min(int(extra_infections), len(susceptible_indices))
                    if num_to_expose > 0:
                        perm = torch.randperm(len(susceptible_indices), device=self.device)[:num_to_expose]
                        selected_indices = susceptible_indices[perm]
                        newly_exposed_today = newly_exposed_today.clone()
                        newly_exposed_today[selected_indices] = 1.0

        daily_infected = daily_infected + (newly_exposed_today.sum() + (extra_infections - num_to_expose)) * time_step_one_hot

        if with_k:
            k_mask = torch.ones_like(daily_infected)
            k_mask[t] = k 
            daily_infected = daily_infected * k_mask

        daily_infected = daily_infected.squeeze(0)

        newly_exposed_today = newly_exposed_today.unsqueeze(1)

        # Disease stages at the start of this timestep (S,E,I,R,D only), before
        # vaccination moves any S -> VACCINATED. Used below so that (a) an agent
        # cannot be both vaccinated and exposed this step and (b) an agent that
        # became R/VACCINATED this step is not eligible to wane this step.
        pre_vaccination_stages = current_stages

        if with_vacc:
            current_stages = self.apply_vaccines(
                current_stages, num_vaccines, t=t,
                agents_ages=agents_ages, priority=vacc_priority,
                essential_first=essential_first,
                household_order=household_order,
            )
            # Vaccination takes precedence over a same-step exposure: only
            # SUSCEPTIBLE agents can become EXPOSED, and these are no longer
            # susceptible.
            just_vaccinated = (
                (pre_vaccination_stages == self.SUSCEPTIBLE_VAR)
                & (current_stages == self.VACCINATED_VAR)
            )
            newly_exposed_today = newly_exposed_today * (~just_vaccinated).float()

        # per-age-group daily NEW exposures (after the vaccinate/expose
        # tie-break). child = U19, adult = 20-64, elderly = 65A. Diagnostic only.
        _ne = newly_exposed_today.reshape(-1)
        _cm, _am, _em = self._age_group_masks(agents_ages)
        self.age_incidence_history.append([
            t,
            float(_ne[_cm].sum().item()),
            float(_ne[_am].sum().item()),
            float(_ne[_em].sum().item()),
        ])

        # per-essentiality-group daily NEW exposures. Logged for EVERY CF type in
        # a work-essentiality run (incl. the CF 3 / CF 7 references, so the
        # analyzer can pair against them) and NEVER otherwise -- so age-priority
        # runs and calibration never touch _require_agent_essential.
        if generating_counterfactual and self.config['simulation_metadata'].get('_essentiality_run', False):
            _em_e, _em_n, _em_o = self._essential_group_masks()
            self.essential_incidence_history.append([
                t,
                float(_ne[_em_e].sum().item()),
                float(_ne[_em_n].sum().item()),
                float(_ne[_em_o].sum().item()),
            ])

        daily_deaths = self.update_number_of_dead(daily_deaths, current_stages, agents_next_stage_times, t, newly_exposed_today)

        updated_stages = self.update_stages(t, current_stages, agents_next_stage_times, newly_exposed_today)
        updated_next_stage_times = self.update_transition_times(
            t, agents_next_stage_times, newly_exposed_today, current_stages
        )

        # --- Configurable immunity waning (R -> S and VACCINATED -> S) -------
        # Runs after the normal S->E->I->R progression. It only moves agents
        # that are in R or VACCINATED *after* this timestep's progression back
        # to S, so the S->E, E->I, I->R logic is untouched. Natural immunity
        # (RECOVERED) and vaccine immunity (VACCINATED) wane on independent
        # durations/rates.
        # instrumentation only (does not affect dynamics): per-step counts of
        # the eligible R / VACCINATED pool and how many of each waned to S.
        n_r_pool = n_r_waned = n_v_pool = n_v_waned = 0.0
        if self.IMMUNITY_WANING_MODE == "fixed":
            # Schedule each transition once, when the agent first enters the
            # state. Do not touch the timer again on later timesteps.
            newly_recovered = (pre_vaccination_stages != self.RECOVERED_VAR) & (
                updated_stages == self.RECOVERED_VAR
            )
            updated_next_stage_times = torch.where(
                newly_recovered,
                torch.full_like(
                    updated_next_stage_times, t + self.RECOVERED_TO_SUSCEPTIBLE_TIME
                ),
                updated_next_stage_times,
            )
            newly_vaccinated = (
                pre_vaccination_stages != self.VACCINATED_VAR
            ) & (updated_stages == self.VACCINATED_VAR)
            updated_next_stage_times = torch.where(
                newly_vaccinated,
                torch.full_like(
                    updated_next_stage_times,
                    t + self.VACCINATED_TO_SUSCEPTIBLE_TIME,
                ),
                updated_next_stage_times,
            )
            # Fire the scheduled transition once the timer is reached. A
            # just-recovered / just-vaccinated agent has timer > t, so it is
            # never waned in the same timestep it enters the state.
            waned = (
                (updated_stages == self.RECOVERED_VAR)
                | (updated_stages == self.VACCINATED_VAR)
            ) & (updated_next_stage_times <= t)
            n_r_pool = float((updated_stages == self.RECOVERED_VAR).sum())
            n_v_pool = float((updated_stages == self.VACCINATED_VAR).sum())
            n_r_waned = float(
                (waned & (updated_stages == self.RECOVERED_VAR)).sum()
            )
            n_v_waned = float(
                (waned & (updated_stages == self.VACCINATED_VAR)).sum()
            )
            updated_stages = torch.where(
                waned,
                torch.full_like(updated_stages, self.SUSCEPTIBLE_VAR),
                updated_stages,
            )
            updated_next_stage_times = torch.where(
                waned,
                torch.full_like(
                    updated_next_stage_times, self.WANING_NO_TRANSITION_TIME
                ),
                updated_next_stage_times,
            )
        elif self.IMMUNITY_WANING_MODE == "stochastic":
            # Only agents already in the protected state at the *start* of this
            # timestep are eligible, so an agent that transitioned I -> R this
            # step, or S -> VACCINATED this step, cannot immediately wane.
            already_recovered = (
                pre_vaccination_stages == self.RECOVERED_VAR
            ) & (updated_stages == self.RECOVERED_VAR)
            already_vaccinated = (
                pre_vaccination_stages == self.VACCINATED_VAR
            ) & (updated_stages == self.VACCINATED_VAR)
            recovered_probability = 1.0 - math.exp(-self.RECOVERED_WANING_RATE)
            vaccinated_probability = 1.0 - math.exp(-self.VACCINATED_WANING_RATE)
            if self._generating_cf():
                _uf = updated_stages.float()
                waning_draw = torch.rand(_uf.shape, generator=self._crn_gen(t, "waning"),
                                         device=_uf.device, dtype=_uf.dtype)
            else:
                waning_draw = torch.rand_like(updated_stages.float())
            r_waned_mask = already_recovered & (waning_draw < recovered_probability)
            v_waned_mask = already_vaccinated & (waning_draw < vaccinated_probability)
            waning_mask = r_waned_mask | v_waned_mask
            n_r_pool = float(already_recovered.sum())
            n_v_pool = float(already_vaccinated.sum())
            n_r_waned = float(r_waned_mask.sum())
            n_v_waned = float(v_waned_mask.sum())
            updated_stages = torch.where(
                waning_mask,
                torch.full_like(updated_stages, self.SUSCEPTIBLE_VAR),
                updated_stages,
            )

        if self.IMMUNITY_WANING_MODE in ("fixed", "stochastic"):
            self.waning_events_history.append(
                [t, n_r_pool, n_r_waned, n_v_pool, n_v_waned]
            )

        updated_infected_times = self.update_infected_times(
            t, agents_infected_time, newly_exposed_today
        )

        self.get_stage_proportions(t, updated_stages)
        self.get_age_stage_proportions(t, updated_stages, agents_ages)
        
        # Log infected commuter stages in Phase 1 final epoch
        is_final_epoch = self.config['simulation_metadata'].get('is_final_epoch', False)
        if metro_phase == 1 and is_final_epoch and hasattr(self, 'commuters') and self.commuters:
            try:
                stages_cpu = updated_stages.cpu().detach().numpy().flatten()
                new_counts = {}
                for agent_id, work_county in self.commuters:
                    if stages_cpu[agent_id] == self.INFECTED_VAR:
                        new_counts[work_county] = new_counts.get(work_county, 0) + 1

                state_prefix = population[:2]
                metro_dir = f"results/{state_prefix}"
                os.makedirs(metro_dir, exist_ok=True)
                csv_path = os.path.join(metro_dir, f"{t}.csv")

                existing_counts = {}
                if os.path.exists(csv_path):
                    try:
                        df_existing = pd.read_csv(csv_path)
                        for _, row in df_existing.iterrows():
                            dest = str(row['destination_county']).zfill(5)
                            num = int(row['num_infected'])
                            existing_counts[dest] = num
                    except Exception:
                        pass

                for dest, num in new_counts.items():
                    existing_counts[dest] = existing_counts.get(dest, 0) + num

                df_new = pd.DataFrame(
                    [{'destination_county': dest, 'num_infected': num} for dest, num in existing_counts.items()],
                    columns=['destination_county', 'num_infected']
                )
                df_new.to_csv(csv_path, index=False)
            except Exception as e:
                print(f"Warning: Failed to log Phase 1 commuter stages: {e}")

        return {
            self.output_variables[0]: updated_stages,
            self.output_variables[1]: updated_next_stage_times,
            self.output_variables[2]: updated_infected_times,
            self.output_variables[3]: daily_infected,
            self.output_variables[4]: daily_deaths,
        }