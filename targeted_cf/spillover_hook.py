"""
Household-spillover secondary-protection analysis for the targeted-allocation CFs.

Output-only analysis. No new CF types, no engine changes: `abm_nets.py`,
`transition.py`, `main.py`, the calibration loop and CF types 1-24 are all
untouched. This module re-runs the *existing* CF simulation (identical params,
seed, networks, waning config -> CRN-paired with the committed CF-A / CF-B run)
and, from the per-step `disease_stage` trajectory that the runner already keeps
in `runner.state_trajectory`, measures the intra-household secondary-protection
signal each CF generates but never reports:

  does a susceptible agent living with a vaccinated household member acquire
  infection at a lower rate than a susceptible agent with no vaccinated
  cohabitant, and does the allocation strategy (elderly-first vs working-age-
  first) change that gap?

Metric: household secondary attack rate (hSAR) in a rolling 14-day window --
the fraction of cohabitants of a vaccinated agent who were susceptible at the
window start and became infected inside the window, versus the same quantity in
households with no vaccinated member. The headline is the per-iteration PAIRED
delta of the protection gap  (hSAR_unvacc - hSAR_vacc)  vs CF7, CRN-paired on
iteration index i, reported mean +/- SD [p10, p90] -- the same discipline as
CF-A / CF-B.

This is a CORRELATIONAL output of the simulation, not a mechanistic within-
household contact path. State it that way in the paper.

Entry point:  python targeted_cf/spillover_hook.py --fips <FIPS> --cf-types 7,12,...
              --n-iters 30 --output-dir results/<FIPS>/TargetedCF/age_priority
              --family age_priority

Fails loud (RuntimeError, non-zero exit) on any missing/short/NaN file, a
Household-ID / num_agents mismatch, an hSAR outside [0, 1], a group-sum
denominator mismatch, or a CF7 case-total regression outside [0.97, 1.03].
"""
import argparse
import glob
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import importlib

import abm_nets
import covid_abm  # noqa: F401  (registers the substeps)
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
from constants import DEVICE
from main import seed_everything, resolve_base_seed

# 6-value age code -> analysis group (identical to transition.py CHILD/ADULT/ELDERLY_CODES)
CHILD_CODES = (5,)
ADULT_CODES = (0, 1, 2, 3)
ELDERLY_CODES = (4,)

WINDOW = 14                     # rolling-window length (days) for the hSAR
CF7_REF = 7                     # reference CF in both families


def die(msg):
    raise RuntimeError(msg)


# --------------------------------------------------------------------------
# household membership + agent-type lookup
# --------------------------------------------------------------------------
class SpilloverHook:
    """Read-only household-membership + age lookups plus the hSAR computation.

    Built once per (county, family); `compute_iteration` is called once per CF
    iteration with that iteration's full [T, N] disease-stage trajectory.
    """

    def __init__(self, fips, num_agents, family, device):
        self.fips = fips
        self.family = family
        self.device = device
        self.N = int(num_agents)

        pop_csv_hits = glob.glob(os.path.join(
            REPO_ROOT, "data", "population_data", "*", f"{fips}_population.csv"))
        if len(pop_csv_hits) != 1:
            die(f"{fips}: expected exactly one committed population csv, found {pop_csv_hits}")
        self.pop_csv = pop_csv_hits[0]
        df = pd.read_csv(self.pop_csv)
        if len(df) != self.N:
            die(f"{self.pop_csv}: {len(df)} rows != num_agents {self.N} (county {fips})")
        if df["Household ID"].isna().any():
            die(f"{self.pop_csv}: 'Household ID' column contains NaN")

        hh_id = df["Household ID"].to_numpy(dtype=np.int64)
        hh_size_col = df["Household Size"].to_numpy(dtype=np.int64)

        groups = defaultdict(list)
        for agent_idx, hid in enumerate(hh_id):
            groups[hid].append(agent_idx)

        seen = np.zeros(self.N, dtype=bool)
        hh_ids_sorted = sorted(groups)
        agent_to_hh = np.full(self.N, -1, dtype=np.int64)
        hh_sizes = np.zeros(len(hh_ids_sorted), dtype=np.int64)
        for hh_idx, hid in enumerate(hh_ids_sorted):
            members = groups[hid]
            hh_sizes[hh_idx] = len(members)
            for m in members:
                if seen[m]:
                    die(f"{self.pop_csv}: agent {m} appears in >1 household")
                seen[m] = True
                agent_to_hh[m] = hh_idx
        if not seen.all():
            die(f"{self.pop_csv}: {int((~seen).sum())} agents not assigned to any household")
        # 'Household Size' column is the declared reference; allow a slack of 1.
        decl = np.array([hh_size_col[groups[hid][0]] for hid in hh_ids_sorted], dtype=np.int64)
        bad = np.where(np.abs(hh_sizes - decl) > 1)[0]
        if bad.size:
            die(f"{self.pop_csv}: {bad.size} households differ from declared 'Household Size' "
                f"by >1 (first: hh_idx {int(bad[0])}, actual {int(hh_sizes[bad[0]])}, "
                f"declared {int(decl[bad[0]])})")
        if hh_sizes.max() > 10:
            die(f"{self.pop_csv}: household of size {int(hh_sizes.max())} > 10 (sanity)")

        self.n_hh = len(hh_ids_sorted)
        self.agent_to_hh = torch.tensor(agent_to_hh, dtype=torch.long, device=device)
        self.hh_size_per_agent = torch.tensor(
            hh_sizes[agent_to_hh], dtype=torch.long, device=device)

        # age masks from the committed age.pickle (same file the sim loads)
        age_pkl = os.path.join(REPO_ROOT, "populations", f"pop{fips}", "age.pickle")
        if not os.path.isfile(age_pkl):
            die(f"age.pickle not found: {age_pkl}")
        age = np.asarray(pd.read_pickle(age_pkl)).reshape(-1)
        if age.shape[0] != self.N:
            die(f"{age_pkl}: {age.shape[0]} ages != num_agents {self.N}")
        a = torch.tensor(age, dtype=torch.long, device=device)
        self.age_group = {
            "child": torch.isin(a, torch.tensor(CHILD_CODES, device=device)),
            "adult": torch.isin(a, torch.tensor(ADULT_CODES, device=device)),
            "elderly": torch.isin(a, torch.tensor(ELDERLY_CODES, device=device)),
        }
        covered = self.age_group["child"] | self.age_group["adult"] | self.age_group["elderly"]
        if not bool(covered.all()):
            die(f"{age_pkl}: {int((~covered).sum())} agents carry an age code outside 0..5")

        self.size_group = {
            "hh1": self.hh_size_per_agent == 1,
            "hh2": self.hh_size_per_agent == 2,
            "hh3p": self.hh_size_per_agent >= 3,
        }

        # disease-stage ids (config-independent constants used by transition.py)
        self.S, self.E, self.VACC = 0, 1, 5

        self.age_cuts = ["child", "adult", "elderly"]
        self.size_cuts = ["hh1", "hh2", "hh3p"]

    # ---- per-iteration hSAR ------------------------------------------------
    def compute_iteration(self, stages):
        """stages: LongTensor [T, N] end-of-day disease stage for one iteration.
        Returns a DataFrame, one row per day, with the hSAR columns."""
        stages = stages.to(self.device)
        T, N = stages.shape
        if N != self.N:
            die(f"compute_iteration: trajectory has {N} agents, expected {self.N}")

        ever_vacc = (stages == self.VACC).cummax(dim=0).values           # [T, N] bool
        became_exposed = torch.zeros_like(stages, dtype=torch.bool)
        became_exposed[1:] = (stages[1:] == self.E) & (stages[:-1] == self.S)

        has_cohab = self.hh_size_per_agent > 1
        rows = []
        for t in range(T):
            w0 = t - (WINDOW - 1)
            if w0 < 0:
                rows.append(self._nan_row(t))
                continue

            susc_at_w0 = stages[w0] == self.S
            vacc_now = ever_vacc[t]
            # household vaccinated-member count as of day t
            hh_vacc = torch.zeros(self.n_hh, dtype=torch.long, device=self.device)
            hh_vacc.scatter_add_(0, self.agent_to_hh, vacc_now.long())
            per_agent_hh_vacc = hh_vacc[self.agent_to_hh]
            has_vacc_cohab = (per_agent_hh_vacc - vacc_now.long()) > 0
            no_vacc_in_hh = per_agent_hh_vacc == 0

            infected_in_window = became_exposed[w0 + 1:t + 1].any(dim=0)

            eligible = susc_at_w0 & has_cohab
            v_grp = {
                "vacc": eligible & has_vacc_cohab,
                "unvacc": eligible & no_vacc_in_hh,
            }

            row = {"t": t}
            # overall + every cut
            self._fill(row, "", v_grp, infected_in_window, None)
            denom_check = {}
            for g in self.age_cuts:
                self._fill(row, f"_{g}", v_grp, infected_in_window, self.age_group[g], denom_check, g)
            for g in self.size_cuts:
                self._fill(row, f"_{g}", v_grp, infected_in_window, self.size_group[g], denom_check, g)

            # group-sum consistency (age groups partition all agents; so do sizes)
            for side in ("vacc", "unvacc"):
                overall = int(v_grp[side].sum())
                age_sum = sum(denom_check[(side, g)] for g in self.age_cuts)
                size_sum = sum(denom_check[(side, g)] for g in self.size_cuts)
                if age_sum != overall:
                    die(f"{self.fips} t={t} {side}: age-group denom sum {age_sum} != overall {overall}")
                if size_sum != overall:
                    die(f"{self.fips} t={t} {side}: hh-size denom sum {size_sum} != overall {overall}")
            rows.append(row)

        df = pd.DataFrame(rows)
        # hSAR must be a probability wherever it is defined
        val_cols = [c for c in df.columns if c.startswith("hSAR_")]
        bad = df[val_cols].to_numpy()
        bad = bad[~np.isnan(bad)]
        if bad.size and (bad.min() < -1e-9 or bad.max() > 1.0 + 1e-9):
            die(f"{self.fips}: hSAR outside [0, 1] (min {bad.min():.4f}, max {bad.max():.4f})")
        return df

    def _nan_row(self, t):
        row = {"t": t, "hSAR_vacc": np.nan, "hSAR_unvacc": np.nan}
        for g in self.age_cuts + self.size_cuts:
            row[f"hSAR_vacc_{g}"] = np.nan
            row[f"hSAR_unvacc_{g}"] = np.nan
        return row

    @staticmethod
    def _fill(row, suffix, v_grp, infected_in_window, sub_mask, denom_check=None, gkey=None):
        for side in ("vacc", "unvacc"):
            m = v_grp[side] if sub_mask is None else (v_grp[side] & sub_mask)
            denom = int(m.sum())
            num = int((m & infected_in_window).sum())
            row[f"hSAR_{side}{suffix}"] = (num / denom) if denom > 0 else np.nan
            if denom_check is not None:
                denom_check[(side, gkey)] = denom


# --------------------------------------------------------------------------
# CF-run driver: re-runs the existing CF simulation, CRN-paired
# --------------------------------------------------------------------------
def _deep_clone_state(state):
    out = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.clone()
        elif isinstance(v, dict):
            out[k] = _deep_clone_state(v)
        else:
            out[k] = v
    return out


def run_with_spillover(fips, cf_types, n_iters, output_dir, family, base_seed):
    """Re-drive the CF loop exactly as abm_nets.eval_net does for the
    counterfactual branch (same seed schedule, same param mapping, same CRN
    keying), but call runner.step() directly and harvest the per-step
    disease_stage trajectory instead of writing cf_raw / plots.

    Writes, per CF type, to <output_dir>/counterfactuals/:
      counterfactual_data<ct>_hh_spillover_raw.csv   (wide, one col-set per iter)
      counterfactual_data<ct>_hh_spillover.csv        (per-day mean)
    Returns {cf_type: {"cf7_total_per_iter": np.array, "exposed_per_iter": np.array}}
    for the regression checks.
    """
    cf_dir = os.path.join(output_dir, "counterfactuals")
    os.makedirs(cf_dir, exist_ok=True)

    applied_seed = seed_everything(base_seed, fips, phase=0, deterministic=True)

    pop_module = importlib.import_module(f"populations.pop{fips}")
    sim = Executor(covid_abm, pop_loader=LoadPopulation(pop_module))
    md = sim.config["simulation_metadata"]
    md["GENERATING_COUNTERFACTUAL"] = True
    md["metro_calibration_phase"] = 0
    md["POPULATION"] = fips
    md["random_seed"] = applied_seed
    md["_essentiality_run"] = False

    original_pop_dir = md["population_dir"]
    md["population_dir"] = os.path.join(os.path.dirname(original_pop_dir), f"pop{fips}")

    runner = sim._get_runner(sim.config)
    runner.init()

    num_steps = int(md["num_steps_per_episode"])
    num_weeks = int(md["NUM_WEEKS"])

    learnable_params = [(name, p) for (name, p) in runner.named_parameters()]

    param_file = os.environ.get("CF_PARAM_FILE")
    if not param_file or not os.path.isfile(param_file):
        die(f"CF_PARAM_FILE not set or missing: {param_file!r}")
    param_array = np.loadtxt(param_file)
    param_tensor = torch.tensor(param_array, dtype=torch.float, device=DEVICE)
    param_tensor = param_tensor[:, None] if param_tensor.ndim == 1 else param_tensor
    if param_tensor.shape[0] < num_weeks + 2:
        die(f"{param_file}: {param_tensor.shape[0]} params, need >= {num_weeks + 2}")
    k_value = float(param_tensor[-1].item())

    initial_state = _deep_clone_state(runner.state)
    hook = SpilloverHook(fips, md["num_agents"], family, DEVICE)

    _uint31 = 2 ** 31 - 1
    diag = {}
    for cf_type in cf_types:
        md["COUNTERFACTUAL_TYPE"] = cf_type
        per_iter_dfs = []
        cf7_total_per_iter, exposed_per_iter = [], []
        for it in range(n_iters):
            md["_cf_iter"] = it
            torch.manual_seed((applied_seed + it) % _uint31)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all((applied_seed + it) % _uint31)

            runner.state = _deep_clone_state(initial_state)
            runner.state_trajectory = []

            abm_nets.map_and_replace_tensor(learnable_params[1][0])(
                runner, True, param_tensor[:num_weeks], mode_calibrate=True)
            abm_nets.map_and_replace_tensor(learnable_params[3][0])(
                runner, True, param_tensor[-2], mode_calibrate=True)
            abm_nets.map_and_replace_tensor(learnable_params[4][0])(
                runner, True, param_tensor[-1], mode_calibrate=True)

            runner.step(num_steps)

            traj = runner.state_trajectory
            stages = torch.stack([
                traj[s][-1]["agents"]["citizens"]["disease_stage"].reshape(-1).long()
                for s in range(len(traj))
            ], dim=0)                                        # [T, N]
            if stages.shape[0] != num_steps:
                die(f"{fips} cf={cf_type} it={it}: trajectory has {stages.shape[0]} steps, "
                    f"expected {num_steps}")
            daily_inf = traj[-1][-1]["environment"]["daily_infected"].reshape(-1).float()
            if daily_inf.numel() != num_steps:
                die(f"{fips} cf={cf_type} it={it}: daily_infected len {daily_inf.numel()} "
                    f"!= {num_steps}")

            eval_start = num_steps - int(md["NUM_WEEKS_TO_EVAL"]) * 7
            cf7_total_per_iter.append(float(daily_inf[eval_start:].sum().item()))
            became_exposed = ((stages[1:] == hook.E) & (stages[:-1] == hook.S)).sum().item()
            exposed_per_iter.append(float(became_exposed))

            df_it = hook.compute_iteration(stages)
            per_iter_dfs.append(df_it)
            print(f"  [{fips}] cf={cf_type} iter={it + 1}/{n_iters} "
                  f"cf_total={cf7_total_per_iter[-1]:.0f}", flush=True)

        _write_spillover_csvs(cf_dir, cf_type, per_iter_dfs)
        diag[cf_type] = dict(
            cf7_total_per_iter=np.array(cf7_total_per_iter),
            exposed_per_iter=np.array(exposed_per_iter),
            k=k_value,
        )
    return diag


def _write_spillover_csvs(cf_dir, cf_type, per_iter_dfs):
    n = len(per_iter_dfs)
    if n == 0:
        die(f"cf={cf_type}: no iterations produced")
    base = per_iter_dfs[0]
    metric_cols = [c for c in base.columns if c != "t"]
    for i, df in enumerate(per_iter_dfs):
        if list(df["t"]) != list(base["t"]):
            die(f"cf={cf_type} iter {i}: day index differs from iter 0")

    wide = pd.DataFrame({"t": base["t"].astype(int)})
    for m in metric_cols:
        for i, df in enumerate(per_iter_dfs):
            wide[f"{m}_{i}"] = df[m].to_numpy()
    raw_path = os.path.join(cf_dir, f"counterfactual_data{cf_type}_hh_spillover_raw.csv")
    wide.to_csv(raw_path, index=False)

    mean_df = pd.DataFrame({"t": base["t"].astype(int)})
    stack = np.stack([df[metric_cols].to_numpy(dtype=float) for df in per_iter_dfs], axis=0)
    with np.errstate(invalid="ignore"):
        mean_df[metric_cols] = np.nanmean(stack, axis=0)
    mean_df.to_csv(os.path.join(cf_dir, f"counterfactual_data{cf_type}_hh_spillover.csv"), index=False)
    print(f"[saved] {raw_path}  ({n} iters, {len(metric_cols)} metrics)", flush=True)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Household-spillover hSAR analysis for the targeted-allocation CFs.")
    ap.add_argument("--fips", required=True)
    ap.add_argument("--cf-types", default="7,12,13,14,15,16,17")
    ap.add_argument("--n-iters", type=int, default=30)
    ap.add_argument("--output-dir", required=True,
                    help="results/<FIPS>/TargetedCF/<TAG>  (same dir as the CF-A/CF-B run)")
    ap.add_argument("--family", default="age_priority",
                    choices=["age_priority", "household_alloc", "essentiality"])
    ap.add_argument("--skip-analyze", action="store_true")
    args = ap.parse_args()

    cf_types = [int(x) for x in args.cf_types.split(",") if x.strip()]
    if CF7_REF not in cf_types:
        die(f"--cf-types must include {CF7_REF} (the paired-delta reference)")

    out = args.output_dir
    existing_cf7 = os.path.join(out, "counterfactuals", f"cf_raw_{CF7_REF}.csv")
    if not os.path.isfile(existing_cf7):
        die(f"{existing_cf7} not found -- run the CF-A (run_age_priority_cf.sh) / CF-B "
            f"(run_essentiality_cf.sh) run first; the spillover analysis reuses its "
            f"network + calibration and CRN-pairs against its CF7.")

    base_seed = resolve_base_seed(None)
    diag = run_with_spillover(args.fips, cf_types, args.n_iters, out, args.family, base_seed)

    # --- mandatory CF7 case-total regression check (section 3.1 / 5.1) -----
    old_cf7 = np.loadtxt(existing_cf7, delimiter=",")
    old_cf7 = old_cf7 if old_cf7.ndim == 2 else old_cf7[None, :]
    old_total = float(old_cf7.sum(axis=1).mean())
    new_total = float(diag[CF7_REF]["cf7_total_per_iter"].mean())
    ratio = new_total / old_total if old_total > 0 else float("nan")
    status = "PASS" if 0.97 <= ratio <= 1.03 else "FAIL"
    print(f"\n[CF7 regression] {args.fips}: hook total {new_total:.0f} vs existing "
          f"{old_total:.0f}  ratio={ratio:.4f}  {status}", flush=True)
    reg_path = os.path.join(out, f"spillover_cf7_regression_{args.fips}.txt")
    with open(reg_path, "w") as fh:
        fh.write(f"fips={args.fips}\nfamily={args.family}\n"
                 f"hook_cf7_total={new_total:.1f}\nexisting_cf7_total={old_total:.1f}\n"
                 f"ratio={ratio:.4f}\nstatus={status}\n")
    if status == "FAIL":
        die(f"{args.fips}: CF7 case total mismatch (ratio {ratio:.4f} outside [0.97, 1.03]). "
            f"SEED / NETWORKS_DIR / CF_PARAM_FILE / vacc-config differ between the committed "
            f"CF run and the spillover run -- CRN pairing is broken.")

    if not args.skip_analyze:
        import analyze_spillover  # noqa
        analyze_spillover.analyze_one(args.fips, cf_types, out, args.family)

    print(f"\n[done] spillover analysis for {args.fips} ({args.family}) -> {out}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print(f"\n[FAIL] {e}\n", file=sys.stderr)
        sys.exit(1)
