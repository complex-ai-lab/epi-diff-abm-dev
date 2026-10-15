import os
import sys
from collections import deque
import numpy as np
from constants import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import json
import pandas as pd

from covid_abm.spline_basis import (
    build_bspline_basis, weekly_midpoint_rows, effective_df, lambda_for_edf,
)


# Human-readable name of each counterfactual policy, keyed by COUNTERFACTUAL_TYPE.
# "factual" = the county's real school/occ intervention series; "flipped" = 1 -
# factual. Used in plot titles and the counterfactual_policies.txt reference.
CF_POLICY_DESC = {
    1:  "all-open (school open, work open)",
    2:  "school-closed, work-open",
    3:  "school-open, work-closed",
    4:  "all-closed (school closed, work closed)",
    5:  "school-open, work-factual",
    6:  "school-open, work-flipped",
    7:  "FACTUAL (factual school, factual work, observed vaccine allocation)",
    8:  "flipped school, factual work",
    9:  "factual school, flipped work",
    10: "flipped school, flipped work",
    11: "factual school, factual work, NO vaccines",
    12: "elderly-first vacc; school factual, work factual",
    13: "working-age-first vacc; school factual, work factual",
    14: "working-age-first vacc; school closed, work open",
    15: "working-age-first vacc; school open, work open",
    16: "elderly-first vacc; school open, work closed",
    17: "elderly-first vacc; school closed, work closed",
    18: "essential-first vacc; occ factual; school factual",
    19: "uniform vacc; non-essential occ closed; school factual",
    20: "essential-first vacc; non-essential occ closed; school factual",
    21: "essential-first vacc; all open",
    22: "uniform vacc; non-essential occ closed; school closed",
    23: "essential-first vacc; non-essential occ closed; school closed",
    24: "uniform vacc; ALL occ closed; school factual (retained-benefit ref)",
    25: "whole-household vacc; random household order; NPI factual",
    26: "whole-household vacc; elderly-home first; NPI factual",
    27: "whole-household vacc; largest households first; NPI factual",
    28: "whole-household vacc; essential-worker homes first; NPI factual",
    29: "UNIFORM-VACC REFERENCE (= the old CF 7); factual school, factual work",
    30: "working-age-first vacc; school closed, factual work",
}

# Folder name per CF type. Shared by execute() and eval_net().
CF_FOLDER_MAP = {
    1: "01_static_all_open",
    2: "02_static_school_closed_work_open",
    3: "03_static_school_open_work_closed",
    4: "04_static_all_closed",
    5: "05_static_school_open_work_factual",
    6: "06_static_school_open_work_flipped",
    7: "07_timevar_factual_school_factual_work",
    8: "08_timevar_flipped_school_factual_work",
    9: "09_timevar_factual_school_flipped_work",
    10: "10_timevar_flipped_school_flipped_work",
    11: "11_timevar_factual_no_vaccines",
    12: "12_agepri_elderly_first_schF_workF",
    13: "13_agepri_workingage_first_schF_workF",
    14: "14_agepri_workingage_first_schC_workO",
    15: "15_agepri_workingage_first_schO_workO",
    16: "16_agepri_elderly_first_schO_workC",
    17: "17_agepri_elderly_first_schC_workC",
    18: "18_ess_essfirst_occfactual_schF",
    19: "19_ess_nonessclose_uniformvacc_schF",
    20: "20_ess_essfirst_nonessclose_schF",
    21: "21_ess_essfirst_allopen",
    22: "22_ess_nonessclose_uniformvacc_schC",
    23: "23_ess_essfirst_nonessclose_schC",
    24: "24_ess_allocc_closed_uniformvacc_schF",
    25: "25_hhalloc_random",
    26: "26_hhalloc_elderly_home_first",
    27: "27_hhalloc_largest_first",
    28: "28_hhalloc_essential_home_first",
    29: "29_uniform_vacc_reference",
    30: "30_agepri_workingage_first_schC_workF",
}


def write_factual_run_csvs(sim, output_dir, param_tensor, case_numbers):
    """After the factual run (CF 7), write three summary CSVs to output_dir:
      beta_daily.csv              day, date, phase, week, beta
      seirv_daily.csv             day, date, S, E, I, R, V   (mean agent counts over CF iterations)
      cases_sim_vs_observed.csv   day, date, simulated_cases, simulated_cases_sd, observed_cases
    `day` is the simulation step t (burn-in included); `date` comes from intervention.csv."""
    md = sim.config['simulation_metadata']
    num_steps = md['num_steps_per_episode']
    num_weeks = md['NUM_WEEKS']
    num_agents = md['num_agents']
    burn_in = int(md.get('BURN_IN_DAYS', 0) or 0)
    report_lag = int(os.environ.get("REPORT_LAG", "0") or 0)
    _bir = os.environ.get("BURN_IN_R", "")
    burn_in_r = float(_bir) if _bir not in ("", "None", "none") else None

    dates = pd.read_csv(f"{md['population_dir']}/intervention.csv").set_index('t')['date']
    days = np.arange(num_steps)

    # 1) beta per day: weekly value from calibrated_params.txt; burn-in days use BURN_IN_R
    R_weekly = param_tensor[:num_weeks].detach().cpu().numpy().ravel()
    week = np.clip((days - burn_in) // 7, 0, num_weeks - 1)
    beta = R_weekly[week]
    if burn_in_r is not None:
        beta = np.where(days < burn_in, burn_in_r, beta)
    pd.DataFrame({
        'day': days, 'date': dates.reindex(days).values,
        'phase': np.where(days < burn_in, 'burn_in', 'observed'),
        'week': np.where(days < burn_in, -1, week), 'beta': beta,
    }).to_csv(os.path.join(output_dir, 'beta_daily.csv'), index=False)

    # 2) S, E, I, R, V counts per day (mean over iterations of the stage fractions x N)
    props = pd.read_csv(os.path.join(output_dir, 'counterfactual_data7_proportions.csv'))
    seirv = pd.DataFrame({'day': props['t'].astype(int)})
    seirv['date'] = dates.reindex(seirv['day']).values
    for col, stage in (('S', 'susceptible'), ('E', 'exposed'), ('I', 'infected'),
                       ('R', 'recovered'), ('V', 'vaccinated')):
        cols = [c for c in props.columns if c == stage or c.startswith(stage + '_')]
        seirv[col] = props[cols].mean(axis=1) * num_agents
    seirv.to_csv(os.path.join(output_dir, 'seirv_daily.csv'), index=False)

    # 3) simulated vs observed reported cases on the scored window
    # (cf_raw_7.csv: one row per iteration, simulation days burn_in .. burn_in + n_eval - 1)
    raw = np.loadtxt(os.path.join(output_dir, 'cf_raw_7.csv'), delimiter=',', ndmin=2)
    n_eval = raw.shape[1]
    sim_days = burn_in + np.arange(n_eval)
    observed = case_numbers.detach().cpu().numpy()[report_lag:report_lag + n_eval]
    pd.DataFrame({
        'day': sim_days, 'date': dates.reindex(sim_days).values,
        'simulated_cases': raw.mean(axis=0), 'simulated_cases_sd': raw.std(axis=0),
        'observed_cases': observed,
    }).to_csv(os.path.join(output_dir, 'cases_sim_vs_observed.csv'), index=False)
    print(f"[INFO] wrote beta_daily.csv, seirv_daily.csv, cases_sim_vs_observed.csv "
          f"({raw.shape[0]} iterations) to {output_dir}", flush=True)


class LearnableParams(nn.Module):
    """Emits [R_1..R_W, infected_proportion, k] from a constant context vector.

    R_SPLINE off (spline_basis=None, the default): the first `num_params`
    head outputs are the weekly R values, squashed into [0, R_MAX]. Historical
    behaviour, bit-for-bit.

    R_SPLINE on: the first `n_basis` head outputs are cubic B-spline
    coefficients for log R, squashed into [log R_MIN, log R_MAX]. The weekly R
    vector is exp(B_w @ coef) where B_w is the design matrix evaluated at the
    weekly midpoints. Because a B-spline basis is a partition of unity,
    log R(t) is a convex combination of the coefficients, so the coefficient
    box transfers exactly to the curve: R stays in [R_MIN, R_MAX] with no
    clamping anywhere. See plans/Task_Spline_Regularization_LogR.md sec 2.

    The two trailing outputs (infection-rate scale, k) keep their original
    boxes in both modes. Those bounds are load-bearing, not cosmetic:
    modify_initial_infected divides by (1 - i0) and k multiplies
    daily_infected as a reporting fraction, so k > 1 is meaningless.
    """

    def __init__(self, num_params, context_dim=3, device=DEVICE,
                 spline_basis=None):
        super().__init__()
        self.device = device
        self.num_params = num_params
        self.spline = spline_basis is not None

        if self.spline:
            B_w, log_r_min, log_r_max = spline_basis
            if B_w.shape[0] != num_params:
                raise RuntimeError(
                    f"LearnableParams: B_w has {B_w.shape[0]} evaluation rows "
                    f"but num_params={num_params} weekly R values are needed")
            self.n_basis = int(B_w.shape[1])
            self.register_buffer(
                "B_w", torch.tensor(B_w, dtype=torch.float32, device=device))
            self.log_r_min = float(log_r_min)
            self.log_r_max = float(log_r_max)
            n_head = self.n_basis + 2
        else:
            self.n_basis = None
            n_head = num_params + 2

        self.fc1 = nn.Linear(context_dim, 64).to(self.device)
        self.fc2 = nn.Linear(64, 32).to(self.device)
        self.fc3 = nn.Linear(32, n_head).to(self.device)
        self.ReLU = nn.ReLU()
        # Unused by forward(), but kept: torch.rand() consumes RNG state, so
        # removing it would shift the random stream and break bit-for-bit
        # reproducibility of every committed R_SPLINE=0 calibration.
        self.learnable_params = nn.Parameter(torch.rand(n_head, device=self.device))

        # Upper bound on the weekly R multipliers. Historically 5.0; env-tunable
        # so the non-identified tail can be capped (see
        # COVID-ABM/Discussion/Regularization_discussion.md 5.7). The two
        # trailing params (infection-rate scale, k) keep their own bounds.
        r_max = float(os.environ.get("R_MAX", "5") or 5.0)
        # Upper bound on the initial infected proportion. Historically 0.0005,
        # i.e. at most ~22 of 43k agents. That is a seed for the START of an
        # epidemic, but the window opens 2020-10-26, ~8 months into the
        # pandemic, with the observed series already at 20-27% of its peak on
        # day 0. Sustaining the observed day-0 incidence needs a PREVALENT pool
        # of I(0) ~ (cases_0 / k) * T_g ~ 240-302 agents, i.e. 0.56-0.70% of the
        # population -- 11-14x what this box allowed. With the bound too low the
        # optimizer buys the level from R instead and drives week-0 R to the top
        # of its own box (01031 saturated it and aborted). See session_summary
        # sections 14 and 16. Env-tunable; default preserves prior behaviour.
        #
        # Note the box is also the initialisation: a sigmoid box starts at its
        # midpoint, so I0_MAX=0.01 starts i0 at 0.005 (~217 agents) instead of
        # 0.00025 (~11). That is a deliberate part of the fix, not a side
        # effect, but it does mean bound and init move together.
        i0_max = float(os.environ.get("I0_MAX", "0.0005") or 0.0005)
        if not (0.0 < i0_max < 1.0):
            raise RuntimeError(f"I0_MAX={i0_max} must lie in (0, 1); "
                               f"modify_initial_infected divides by (1 - i0)")
        min_values_list = [0] * num_params + [0.0] + [0.2]
        max_values_list = [r_max] * num_params + [i0_max] + [1]

        self.min_values = torch.tensor(min_values_list, device=self.device)
        self.max_values = torch.tensor(max_values_list, device=self.device)
        # Same two trailing bounds, sliced out for the spline path.
        self.tail_min = self.min_values[-2:]
        self.tail_max = self.max_values[-2:]

        # --- matched initialisation for the spline path ----------------------
        # The two parameterizations do NOT start at the same R, because a
        # sigmoid box is linear in its own output:
        #   raw box:  R     = R_MAX * sigmoid(z)      -> R = R_MAX/2 at z=0
        #   log box:  log R = lo + W * sigmoid(z)     -> R = sqrt(R_MIN*R_MAX)
        # i.e. the log box's midpoint is the GEOMETRIC mean of the box, so with
        # R_MIN=0.1, R_MAX=5 the spline starts at R=0.707 while the default path
        # starts at R=2.5 -- a 3.54x gap that made the first Stage 1 run vary
        # the basis and the initialisation together. Since the D2 penalty's null
        # space is exactly the constant and linear coefficient sequences,
        # nothing pulls the LEVEL of log R back up, so the run settles low on
        # the R/k ridge.
        #
        # R_SPLINE_INIT_R shifts the coefficient block's bias so E[R] at init is
        # the requested value. Unset -> untouched, i.e. the historical spline
        # behaviour. The shift is in-place under no_grad and draws no random
        # numbers, so the RNG stream is unaffected either way.
        self.init_r = None
        _init_r = os.environ.get("R_SPLINE_INIT_R", "").strip()
        if self.spline and _init_r:
            tgt = float(_init_r)
            r_lo, r_hi = float(np.exp(self.log_r_min)), float(np.exp(self.log_r_max))
            if not (r_lo < tgt < r_hi):
                raise RuntimeError(
                    f"R_SPLINE_INIT_R={tgt} must lie strictly inside the "
                    f"coefficient box ({r_lo:.4f}, {r_hi:.4f})")
            s = (float(np.log(tgt)) - self.log_r_min) / (self.log_r_max - self.log_r_min)
            z = float(np.log(s / (1.0 - s)))
            with torch.no_grad():
                self.fc3.bias[:self.n_basis] += z
            self.init_r = tgt

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(self.device)
        out = self.ReLU(self.fc1(x))
        out = self.ReLU(self.fc2(out))
        out = self.fc3(out)

        if not self.spline:
            params = self.min_values + (self.max_values -
                                        self.min_values) * self.sigmoid(out)
            return params, None

        coef = self.log_r_min + (self.log_r_max - self.log_r_min) * \
            self.sigmoid(out[:self.n_basis])
        R_weekly = torch.exp(self.B_w @ coef)                 # [num_params]
        tail = self.tail_min + (self.tail_max -
                                self.tail_min) * self.sigmoid(out[self.n_basis:])
        return torch.cat([R_weekly, tail]), coef

def map_and_replace_tensor(input_string):
    parts = input_string.split('.')
    function = parts[1]
    index = parts[2]
    sub_func = parts[3]
    var_name = parts[5]

    def getter_and_setter(runner, counterfactual=False, new_value=None, mode_calibrate=True):
        substep_type = getattr(runner.initializer, function)
        substep_function = getattr(substep_type[str(index)], sub_func)

        if mode_calibrate:
            current_tensor = getattr(substep_function, 'calibrate_' + var_name)
        else:
            current_tensor = getattr(getattr(substep_function, 'learnable_args'), var_name)
        
        if new_value is not None:
            if not counterfactual:
                assert new_value.requires_grad == current_tensor.requires_grad
            if mode_calibrate:
                setvar_name = 'calibrate_' + var_name
                setattr(substep_function, setvar_name, new_value)
                current_tensor = getattr(substep_function, setvar_name)
            else:
                setvar_name = var_name
                subfunc_param = getattr(substep_function, 'learnable_args')
                setattr(subfunc_param, setvar_name, new_value)
                current_tensor = getattr(subfunc_param, setvar_name)

            return current_tensor
        else:
            return current_tensor

    return getter_and_setter

def execute(sim, runner, Y_actual, epoch, epochs, n_steps=28, loss_cutoff_step=None):
    runner.step(n_steps)

    labels = runner.state_trajectory[-1][-1]['environment']['daily_infected']

    weeks_to_eval = sim.config['simulation_metadata']['NUM_WEEKS_TO_EVAL']

    # --- simulation / observation register -----------------------------------
    # daily_infected[t] is k x the number of agents newly EXPOSED on day t
    # (transition.py:1176-1181). No incubation, test-seeking or reporting delay
    # is modelled anywhere between that exposure and the number compared against
    # data; k is a scalar reporting FRACTION, not a delay.
    #
    # intervention.csv is selected by `t` (transition.py:897) and its row t=0 is
    # the same calendar date as row 0 of daily_data.csv, so simulation day 0 and
    # observation day 0 are the SAME DAY and the two series align head to head.
    #
    # The previous code took the LAST weeks_to_eval*7 simulated days and scored
    # them against the observation series from its row 0. With n_steps=182 and
    # 168 observed days that paired simulation day 14+i with observation day i:
    # it asked the model to reproduce, 14 days AFTER a date, the cases reported
    # ON it. That is anti-causal -- a real reporting lag makes cases FOLLOW
    # infections -- and because R is indexed by int(t/7), 14 days is exactly two
    # R-indices, so the calibration could absorb it by relabelling R_w -> R_w+2
    # while intervention.csv stayed pinned to true calendar dates. The learned
    # R(t) ended up two weeks out of phase with the interventions it is supposed
    # to be separable from. See session_summary.md sections 9.3 and 11.1-11.2.
    #
    # REPORT_LAG (days, default 0) is the honest form of an offset here:
    #     actual[d + L]  ~  k * E_new(d)  =  labels[d]
    # so labels[:n] is scored against Y_actual[L:L+n]. L=0 keeps every observed
    # day and makes no modelling claim beyond correcting the register; L in
    # 7..14 is the epidemiologically defensible range and is left as an explicit
    # sweep rather than a silent default.
    report_lag = int(os.environ.get("REPORT_LAG", "0") or 0)
    if report_lag < 0:
        raise ValueError(f"REPORT_LAG must be >= 0, got {report_lag}")

    # Unscored burn-in days before the observation window. Simulation day
    # burn_in + d is observation day d, because intervention.csv was shifted
    # back by the same amount (scripts/add_burnin_days.py), so R(t) stays in
    # phase with the NPIs and doses.
    burn_in = int(sim.config['simulation_metadata'].get('BURN_IN_DAYS', 0) or 0)

    n_eval = min(n_steps - burn_in - report_lag, len(Y_actual) - report_lag,
                 weeks_to_eval * 7)
    if n_eval <= 0:
        raise RuntimeError(
            f"empty evaluation window: n_steps={n_steps}, "
            f"len(Y_actual)={len(Y_actual)}, NUM_WEEKS_TO_EVAL={weeks_to_eval}, "
            f"BURN_IN_DAYS={burn_in}, REPORT_LAG={report_lag}")

    labels = labels[burn_in:burn_in + n_eval]
    Y_actual = Y_actual[report_lag:report_lag + n_eval]
    Y_sched = labels.to(Y_actual.device)

    date = sim.config['simulation_metadata']['DATE']
    initial_rate = sim.config['simulation_metadata']['INITIAL_INFECTION_RATE']
    exposed_to_infected = sim.config['simulation_metadata']['EXPOSED_TO_INFECTED_TIME']
    infected_to_recovered = sim.config['simulation_metadata']['INFECTED_TO_RECOVERED_TIME']
    population = sim.config['simulation_metadata']['POPULATION']
    generating_counterfactual = sim.config['simulation_metadata']['GENERATING_COUNTERFACTUAL']
    cf_type = sim.config['simulation_metadata']['COUNTERFACTUAL_TYPE']
    with_k = sim.config['simulation_metadata']['WITH_K']
    with_vacc = sim.config['simulation_metadata']['WITH_VACC']
    use_7day_avg = sim.config['simulation_metadata'].get('USE_7DAY_AVG', True)

    metro_phase = sim.config['simulation_metadata'].get('metro_calibration_phase', 0)
    if generating_counterfactual:
        labels_np = labels.cpu().detach().numpy()
        # CF_OUTPUT_DIR redirects every counterfactual artifact away from
        # result_graphs/... so an existing calibration/CF result is never
        # overwritten. Unset -> original behaviour.
        cf_out = os.environ.get("CF_OUTPUT_DIR")
        output_dir = cf_out or f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}_{use_7day_avg}_metro_{metro_phase}'
        os.makedirs(output_dir, exist_ok=True)

        cf_folder_name = CF_FOLDER_MAP.get(cf_type, f"type_{cf_type}")
        central_root = os.path.join(cf_out, "all_counterfactual_results") if cf_out else "all_counterfactual_results"
        central_graph_dir = os.path.join(central_root, cf_folder_name, "graphs")
        central_data_dir = os.path.join(central_root, cf_folder_name, "data")
        os.makedirs(central_graph_dir, exist_ok=True)
        os.makedirs(central_data_dir, exist_ok=True)

        # Plot comparison if possible
        try:
            county_data = pd.read_csv(f"data/processed_data/{population}/{date}/daily_data.csv", parse_dates=['date'])
            case_col = 'cases_7day_avg' if use_7day_avg else 'cases_singular'
            if case_col not in county_data.columns:
                case_col = 'cases'
            county_cases = torch.tensor(county_data[case_col].values[:n_steps], dtype=torch.float)
            county_cases_np = county_cases.cpu().detach().numpy()

            plt.figure(figsize=(8, 6))
            plt.plot(labels_np, marker='o', label='Simulation Cases')
            plt.plot(county_cases_np, marker='x', label='Actual Cases')
            plt.xlabel('Days')
            plt.ylabel('Number of Cases')
            plt.title(f'CF Type {cf_type} ({CF_POLICY_DESC.get(cf_type, "?")}) '
                      f'— {population}\nFactual vs Counterfactual daily cases')
            plt.legend()
            plt.savefig(f'{output_dir}/counterfactual_results{cf_type}.png')
            plt.savefig(os.path.join(central_graph_dir, f"{population}_counterfactual_results{cf_type}.png"))
            plt.close()

            counterfactual_df = pd.DataFrame({
                'day': range(len(labels_np)),
                'counterfactual_cases': labels_np,
                'actual_cases': county_cases_np[:len(labels_np)]
            })
        except:
            print("Warning: Could not load factual data for comparison plotting.")
            counterfactual_df = pd.DataFrame({
                'day': range(len(labels_np)),
                'counterfactual_cases': labels_np
            })

        counterfactual_df.to_csv(f'{output_dir}/counterfactual_data{cf_type}.csv', index=False)
        counterfactual_df.to_csv(os.path.join(central_data_dir, f"{population}_counterfactual_data{cf_type}.csv"), index=False)

        # Append mode for raw infected labels
        file_path_raw = os.path.join(output_dir, f'cf_raw_{cf_type}.csv')
        with open(file_path_raw, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(labels_np.tolist())

        return 0.0

    if epoch % 50 == 0 or epoch == epochs - 1:
        county_data = pd.read_csv(f"data/processed_data/{population}/{date}/daily_data.csv", parse_dates=['date'])
        case_col = 'cases_7day_avg' if use_7day_avg else 'cases_singular'
        if case_col not in county_data.columns:
            case_col = 'cases'
        county_cases = torch.tensor(county_data[case_col].values[:n_steps], dtype=torch.float)
        
        labels_np = labels.cpu().detach().numpy()
        county_cases_np = county_cases.cpu().detach().numpy()
        
        plt.figure(figsize=(8, 6))
        plt.plot(labels_np, marker='o', label='Simulation Cases')
        plt.plot(county_cases_np, marker='x', label='Actual Cases')
        plt.xlabel('Days')
        plt.ylabel('Number of Cases')
        plt.title(f'Simulation vs Actual Cases in {population}')
        plt.legend()
        
        metro_phase = sim.config['simulation_metadata'].get('metro_calibration_phase', 0)
        os.makedirs(f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}_{use_7day_avg}_metro_{metro_phase}/{epoch}', exist_ok=True)
        plt.savefig(f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}_{use_7day_avg}_metro_{metro_phase}/{epoch}/simulation_results.png')
        plt.close()

        trans_substep = runner.initializer.transition_function['0'].new_transmission
        history_data = trans_substep.proportion_history
        if history_data:
            df = pd.DataFrame(history_data, columns=["t", "susceptible", "exposed", "infected", "recovered", "dead"])
            stages = ["susceptible", "exposed", "infected", "recovered", "dead"]
            for stage in stages:
                plt.figure()
                plt.plot(df["t"], df[stage], marker='o', label=stage)
                plt.xlabel("Time (t)")
                plt.ylabel(f"{stage.capitalize()} Proportion")
                plt.title(f"{stage.capitalize()} Proportion Over Time")
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}_{use_7day_avg}_metro_{metro_phase}/{epoch}/{stage}_proportion.png")
                plt.close()

    if epoch == epochs - 1:
        metro_phase = sim.config['simulation_metadata'].get('metro_calibration_phase', 0)
        output_dir = f'results/{population}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_metro_{metro_phase}'
        os.makedirs(output_dir, exist_ok=True)
        generated_df = pd.DataFrame({
            "day": list(range(len(Y_sched))),
            "generated_factual_cases": Y_sched.cpu().detach().numpy()
        })
        generated_df.to_csv(f"{output_dir}/generated_factual.csv", index=False)

    if loss_cutoff_step is not None:
        Y_sched_eval = Y_sched[:loss_cutoff_step]
        Y_actual_eval = Y_actual[:loss_cutoff_step]
    else:
        Y_sched_eval = Y_sched
        Y_actual_eval = Y_actual

    # --- data-fit loss -------------------------------------------------------
    # LOSS_MODE selects how the daily-case residual is scored. Default "sse"
    # leaves the historical behaviour byte-for-byte unchanged.
    #   sse     : summed squared error on raw counts (mean * N). Peak weeks
    #             (~100+ cases) dominate the gradient; low-count plateau / tail
    #             weeks (2-9 cases) contribute ~1/1000 as much, so tail R is
    #             effectively unconstrained by the data (see
    #             COVID-ABM/Discussion/Regularization_discussion.md sec 1.4).
    #   log1p   : squared error on log(1 + cases) -> relative errors, every
    #             week contributes on the same scale. Recovers the mid-window
    #             plateau and gives the tail real gradient.
    #   poisson : Poisson NLL  sum(lam - y*log lam)  with lam = sim cases.
    #             Variance grows with the mean, so again a relative-error
    #             weighting, with the count-data noise model.
    LOSS_MODE = os.environ.get("LOSS_MODE", "sse").strip().lower()
    diff2 = torch.clamp((Y_actual_eval - Y_sched_eval) ** 2, min=0)
    if LOSS_MODE == "sse":
        total_loss = diff2.mean() * len(diff2)
    elif LOSS_MODE == "log1p":
        d = torch.log1p(Y_sched_eval.clamp_min(0.0)) - torch.log1p(Y_actual_eval.clamp_min(0.0))
        total_loss = (d ** 2).mean() * len(d)
    elif LOSS_MODE == "poisson":
        lam = Y_sched_eval.clamp_min(1e-6)
        total_loss = (lam - Y_actual_eval * torch.log(lam)).mean() * len(lam)
    else:
        raise ValueError(f"unknown LOSS_MODE={LOSS_MODE!r} (expected sse|log1p|poisson)")

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"[loss] mode={LOSS_MODE} epoch={epoch} "
              f"data_loss={float(total_loss.detach().cpu()):.4f} "
              f"(sse-equiv {float(diff2.mean() * len(diff2)):.1f})", flush=True)

    return total_loss

def eval_net(sim, runner):
    learnable_params = [(name, param) for (name, param) in runner.named_parameters()]
    
    date = sim.config['simulation_metadata']['DATE']
    num_steps = sim.config['simulation_metadata']['num_steps_per_episode']
    num_weeks = sim.config['simulation_metadata']['NUM_WEEKS']
    population = sim.config['simulation_metadata']['POPULATION']
    raw_cf = sim.config['simulation_metadata'].get('GENERATING_COUNTERFACTUAL', False)
    generating_counterfactual = str(raw_cf).lower() in ['true', '1']
    print(f"[INFO] Mode: GENERATING_COUNTERFACTUAL = {generating_counterfactual}", flush=True)
    initial_rate = sim.config['simulation_metadata']['INITIAL_INFECTION_RATE']
    exposed_to_infected = sim.config['simulation_metadata']['EXPOSED_TO_INFECTED_TIME']
    infected_to_recovered = sim.config['simulation_metadata']['INFECTED_TO_RECOVERED_TIME']
    with_k = sim.config['simulation_metadata']['WITH_K']
    with_vacc = sim.config['simulation_metadata']['WITH_VACC']
    use_7day_avg = sim.config['simulation_metadata'].get('USE_7DAY_AVG', True)

    df = pd.read_csv(f"data/processed_data/{population}/{date}/daily_data.csv", parse_dates=["date"])
    case_col = 'cases_7day_avg' if use_7day_avg else 'cases_singular'
    if case_col not in df.columns:
        case_col = 'cases'
    case_numbers = torch.tensor(df[case_col].values, dtype=torch.float, device=DEVICE)

    context_tensor = torch.tensor([initial_rate, exposed_to_infected, infected_to_recovered], dtype=torch.float, device=DEVICE)

    # --- P-spline parameterization of log R(t) (Stage 1) ---------------------
    # R_SPLINE=1 swaps the 26 free weekly log-R levels for a cubic B-spline on
    # log R, evaluated at the weekly midpoints so the ABM still receives the
    # same [26, 1] tensor and calibrated_params.txt keeps its 28-row contract.
    # Smoothing is judged by effective degrees of freedom (EDF), not by the
    # penalty-to-data ratio, which saturates around 0.006 at any lambda.
    # See plans/Task_Spline_Regularization_LogR.md sections 1.5 and 2.
    R_SPLINE = os.environ.get("R_SPLINE", "0").strip().lower() in ("1", "true", "yes")
    R_SPLINE_J = int(os.environ.get("R_SPLINE_J", str(num_weeks)) or num_weeks)
    R_SPLINE_DEGREE = int(os.environ.get("R_SPLINE_DEGREE", "3") or 3)
    R_SPLINE_LAMBDA = float(os.environ.get("R_SPLINE_LAMBDA", "5") or 5.0)
    R_SPLINE_END_LAMBDA = float(
        os.environ.get("R_SPLINE_ENDPOINT_LAMBDA", "5") or 5.0)
    R_SPLINE_R_MIN = float(os.environ.get("R_SPLINE_R_MIN", "0.1") or 0.1)
    R_SPLINE_R_MAX = float(os.environ.get("R_MAX", "5") or 5.0)

    spline_basis = None
    B_w_np = D2_np = D2_t = None
    spline_edf = float("nan")
    if R_SPLINE:
        if float(os.environ.get("R_SMOOTH_LAMBDA", "0") or 0.0) > 0:
            raise RuntimeError(
                "R_SPLINE=1 and R_SMOOTH_LAMBDA>0 are mutually exclusive: the "
                "spline penalty replaces the step-function RW penalty. "
                "Set R_SMOOTH_LAMBDA=0.")
        # The basis spans the OBSERVATION window (num_steps - burn-in), not the
        # burn-in, so r_daily.txt keeps its 182 rows on the same calendar as the
        # pre-burn-in runs and the R(t) plots stay directly comparable. Burn-in
        # days run at week 0 R (transition.py week_id).
        _burn_in = int(sim.config['simulation_metadata'].get('BURN_IN_DAYS', 0) or 0)
        _obs_days = num_steps - _burn_in
        # R_SPLINE_SCORED_ONLY: build the basis over the SCORED window only.
        #
        # Why. NUM_WEEKS=26 parameterises R over 182 observation days, but
        # daily_data.csv holds 168 (2020-10-26..2021-04-11), so
        # n_eval = min(182, 168, 24*7) = 168 and weeks 24-25 are never compared
        # against anything. Those coefficients are then shaped only by the
        # penalty -- and because a cubic B-spline coefficient spans ~4 knot
        # intervals (~28 days), they also carry 99.9% / 80.7% / 19.9% of the R
        # at scored weeks 23 / 22 / 21. So the free tail reaches ~3 weeks back
        # into scored territory and drags the last scored weeks up with it:
        # 36003 overshoots week 23 by 2.4x, 01031 by 3.0x.
        # See session_summary.md 19.1 and 19.3.
        #
        # With this on, the basis spans the 168 scored days with J = 24
        # coefficients evaluated at the 24 scored week midpoints. Weeks 24-25
        # are then set EXPLICITLY to week 23's value by duplicating the last
        # evaluation row, so they add no free parameters, and the 26 weekly R
        # values -- hence the 28-row calibrated_params.txt contract -- are
        # unchanged. The duplicated rows are excluded from the EDF and lambda
        # calculations, which must see one row per independent observation.
        _scored_only = str(os.environ.get("R_SPLINE_SCORED_ONLY", "0")).strip().lower() \
            in ("1", "true", "yes")
        if _scored_only:
            _weeks_to_eval = int(
                sim.config['simulation_metadata']['NUM_WEEKS_TO_EVAL'])
            _n_scored = min(_obs_days, len(case_numbers), _weeks_to_eval * 7)
            _spline_weeks = _n_scored // 7
            _spline_days = _spline_weeks * 7
            if _spline_weeks < 2 or _spline_weeks > num_weeks:
                raise RuntimeError(
                    f"R_SPLINE_SCORED_ONLY: derived {_spline_weeks} scored "
                    f"weeks from n_scored={_n_scored}, num_weeks={num_weeks}")
            R_SPLINE_J = int(os.environ.get("R_SPLINE_J", str(_spline_weeks))
                             or _spline_weeks)
            # More coefficients than scored evaluation rows leaves the fit
            # underdetermined before the penalty is even applied. The caller
            # must drop J to the number of scored weeks (the driver does).
            if R_SPLINE_J > _spline_weeks:
                raise RuntimeError(
                    f"R_SPLINE_SCORED_ONLY: R_SPLINE_J={R_SPLINE_J} exceeds "
                    f"the {_spline_weeks} scored evaluation rows. Set "
                    f"R_SPLINE_J={_spline_weeks} (or lower).")
        else:
            _spline_days = _obs_days
            _spline_weeks = num_weeks
        B_np, D2_np = build_bspline_basis(_spline_days, R_SPLINE_J, R_SPLINE_DEGREE)
        B_w_scored = B_np[weekly_midpoint_rows(_spline_days, _spline_weeks)]  # [Ws, J]
        if _spline_weeks < num_weeks:
            B_w_np = np.vstack([
                B_w_scored,
                np.repeat(B_w_scored[-1:], num_weeks - _spline_weeks, axis=0),
            ])                                                          # [W, J]
        else:
            B_w_np = B_w_scored
        _edf_target = os.environ.get("R_SPLINE_EDF", "").strip()
        if _edf_target:
            R_SPLINE_LAMBDA = lambda_for_edf(B_w_scored, D2_np, float(_edf_target))
            print(f"[INFO] R_SPLINE_EDF={_edf_target} -> "
                  f"lambda={R_SPLINE_LAMBDA:.6f}", flush=True)
        spline_basis = (B_w_np, np.log(R_SPLINE_R_MIN), np.log(R_SPLINE_R_MAX))
        D2_t = torch.tensor(D2_np, dtype=torch.float32, device=DEVICE)
        spline_edf = effective_df(B_w_scored, D2_np, R_SPLINE_LAMBDA)
        if _scored_only:
            print(f"[INFO] R_SPLINE_SCORED_ONLY: basis over {_spline_days} "
                  f"scored days, {_spline_weeks} weeks, J={R_SPLINE_J}; weeks "
                  f"{_spline_weeks}-{num_weeks - 1} pinned to week "
                  f"{_spline_weeks - 1}", flush=True)
        _init_r_env = os.environ.get("R_SPLINE_INIT_R", "").strip()
        _init_r_desc = _init_r_env if _init_r_env else (
            f"{np.sqrt(R_SPLINE_R_MIN * R_SPLINE_R_MAX):.3f} "
            f"(unset -> geometric box midpoint)")
        print(f"[INFO] R spline ON: J={R_SPLINE_J} degree={R_SPLINE_DEGREE} "
              f"lambda={R_SPLINE_LAMBDA:.6f} "
              f"endpoint_lambda={R_SPLINE_END_LAMBDA} "
              f"log-R box=[{np.log(R_SPLINE_R_MIN):.3f},"
              f"{np.log(R_SPLINE_R_MAX):.3f}] EDF={spline_edf:.4f} "
              f"cond(B_w)={np.linalg.cond(B_w_np):.3e} "
              f"init_R={_init_r_desc}", flush=True)
        if not (2.5 <= spline_edf <= R_SPLINE_J - 0.5):
            raise RuntimeError(
                f"R_SPLINE_LAMBDA={R_SPLINE_LAMBDA} gives EDF={spline_edf:.3f} "
                f"at J={R_SPLINE_J}. EDF<=2.5 is a straight line in log R; "
                f"EDF>=J-0.5 means the penalty is inactive. Choose lambda via "
                f"R_SPLINE_EDF=<target>.")

    learn_model = LearnableParams(num_weeks, context_dim=3, device=DEVICE,
                                  spline_basis=spline_basis)

    metro_phase = sim.config['simulation_metadata'].get('metro_calibration_phase', 0)
    if metro_phase == 2:
        phase1_results_dir = f'results/{population}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_metro_1'
        model_path = os.path.join(phase1_results_dir, "calibrated_model.pt")
        if os.path.exists(model_path):
            print(f"Loading Phase 1 weights from {model_path}...")
            learn_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        else:
            print(f"Warning: Phase 1 weights not found at {model_path}. Starting calibration from scratch.")
            
    learn_model = torch.compile(learn_model)

    def deep_clone_state(state):
        new_state = {}
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                new_state[k] = v.clone()
            elif isinstance(v, dict):
                new_state[k] = deep_clone_state(v)
            else:
                new_state[k] = v
        return new_state
    
    initial_state = deep_clone_state(runner.state)

    metro_phase = sim.config['simulation_metadata'].get('metro_calibration_phase', 0)
    if generating_counterfactual:
        # CF_TYPES / CF_ITERS: optional overrides for smoke tests (default: all
        # 11 types, 30 iterations).
        _cf_types_env = os.environ.get("CF_TYPES", "").strip()
        cf_types_to_run = ([int(x) for x in _cf_types_env.split(",") if x]
                           if _cf_types_env else range(1, 29))
        _cf_iters_override = int(os.environ["CF_ITERS"]) if os.environ.get("CF_ITERS") else None
        cf_folder_map = CF_FOLDER_MAP

        param_load_phase = 0 if metro_phase == 0 else 2
        base_dir = f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}_{use_7day_avg}_metro_{param_load_phase}'
        # CF_PARAM_FILE lets a CF run use a specific frozen calibration
        # (e.g. a waning-mode fit) without staging it into result_graphs/...
        param_file = os.environ.get("CF_PARAM_FILE") or os.path.join(base_dir, "calibrated_params.txt")

        # BETA / INITIAL_INFECTED / K in config.yaml (exported by main.py as
        # BETA_OVERRIDE / I0_OVERRIDE / K_OVERRIDE, JSON) replace the calibrated
        # values. With all three set, no calibration is needed.
        overrides = {name: json.loads(os.environ[name])
                     for name in ("BETA_OVERRIDE", "I0_OVERRIDE", "K_OVERRIDE")
                     if os.environ.get(name, "").strip() not in ("", "null", "None")}

        if os.path.exists(param_file):
            try:
                param_array = np.loadtxt(param_file)
            except Exception as e:
                print(f"\n[ERROR] Failed to load calibrated parameters from '{param_file}': {e}\n")
                sys.exit(1)
        elif len(overrides) == 3:
            param_array = np.zeros(num_weeks + 2)
        else:
            print(f"\n[ERROR] Calibrated parameter file not found at: '{param_file}'")
            print(f"[ERROR] Run calibrate.sh first, or set all of BETA, INITIAL_INFECTED and K in config.yaml.\n")
            sys.exit(1)

        param_array = np.asarray(param_array, dtype=float).ravel()
        if "BETA_OVERRIDE" in overrides:
            beta = np.atleast_1d(np.asarray(overrides["BETA_OVERRIDE"], dtype=float))
            if beta.size == 1:
                beta = np.repeat(beta, num_weeks)
            if beta.size != num_weeks:
                raise ValueError(f"BETA must be one number or a list of {num_weeks} weekly values, got {beta.size}")
            param_array[:num_weeks] = beta
        if "I0_OVERRIDE" in overrides:
            param_array[-2] = float(overrides["I0_OVERRIDE"])
        if "K_OVERRIDE" in overrides:
            param_array[-1] = float(overrides["K_OVERRIDE"])
        if overrides:
            print(f"[INFO] parameter overrides from config: {sorted(overrides)}; "
                  f"beta mean {param_array[:num_weeks].mean():.4f}, "
                  f"initial infected {param_array[-2]:.6f}, k {param_array[-1]:.4f}", flush=True)

        param_tensor = torch.tensor(param_array, dtype=torch.float, device=DEVICE)
        param_tensor = param_tensor[:, None] if param_tensor.ndim == 1 else param_tensor

        # A work-essentiality run: any of CF 18-23 present -> log the per-
        # essentiality-group incidence for EVERY CF in the run (incl. 3/7 refs).
        _ess_run = any(int(ct) in (18, 19, 20, 21, 22, 23, 24) for ct in cf_types_to_run)
        sim.config['simulation_metadata']['_essentiality_run'] = _ess_run

        for cf_type in cf_types_to_run:
            sim.config['simulation_metadata']['COUNTERFACTUAL_TYPE'] = cf_type
            cf_name = cf_folder_map.get(cf_type, f"type_{cf_type}")
            print(f"\n=======================================================", flush=True)
            print(f"Running Counterfactual Type {cf_type} ({cf_name}) for FIPS {population}", flush=True)
            print(f"=======================================================", flush=True)

            all_age_proportions = []
            all_age_incidence = []
            all_essential_incidence = []
            all_layer_attribution = []
            _ess_cf = _ess_run
            num_iterations = _cf_iters_override or 30

            _cf_base_seed = int(sim.config['simulation_metadata'].get('random_seed', 42))
            for num in range(num_iterations):
                # Common random numbers: iteration `num` uses the same RNG state
                # for every CF type. The substeps key their stochastic draws on
                # `_cf_iter` (+ step + call-site), and this resets the global
                # stream as a backstop for anything not routed through a
                # dedicated generator.
                sim.config['simulation_metadata']['_cf_iter'] = num
                torch.manual_seed((_cf_base_seed + num) % (2**31 - 1))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all((_cf_base_seed + num) % (2**31 - 1))

                runner.state = deep_clone_state(initial_state)
                runner.state_trajectory = []

                tensorfunc_R = map_and_replace_tensor(learnable_params[1][0])
                tensorfunc_R(runner, True, param_tensor[:num_weeks], mode_calibrate=True)
                
                tensorfunc_inf = map_and_replace_tensor(learnable_params[3][0])
                tensorfunc_inf(runner, True, param_tensor[-2], mode_calibrate=True)

                tensorfunc_k = map_and_replace_tensor(learnable_params[4][0])
                tensorfunc_k(runner, True, param_tensor[-1], mode_calibrate=True)

                _ = execute(sim, runner, case_numbers, epoch=0, epochs=1, n_steps=num_steps)

                substep = runner.initializer.transition_function['0'].new_transmission
                output_dir = os.environ.get("CF_OUTPUT_DIR") or f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}_{use_7day_avg}_metro_{param_load_phase}'
                os.makedirs(output_dir, exist_ok=True)
                path = f'{output_dir}/counterfactual_data{cf_type}_proportions.csv'

                _it = num if num_iterations > 1 else 0
                if num_iterations > 1:
                    substep.save_proportions_to_disk2(path, num)
                else:
                    substep.save_proportions_to_disk(path)
                substep.save_age_incidence_to_disk(
                    f'{output_dir}/counterfactual_data{cf_type}_age_incidence_raw.csv', _it)
                substep.save_vacc_by_age_to_disk(
                    f'{output_dir}/counterfactual_data{cf_type}_vacc_by_age_raw.csv', _it)
                if _ess_cf:
                    substep.save_essential_incidence_to_disk(
                        f'{output_dir}/counterfactual_data{cf_type}_essential_incidence_raw.csv', _it)
                if getattr(substep, "layer_attribution", False):
                    substep.save_layer_attribution_to_disk(
                        f'{output_dir}/counterfactual_data{cf_type}_layer_attribution_raw.csv', _it)

                all_age_proportions.append(substep.age_proportion_history)
                all_age_incidence.append(substep.age_incidence_history)
                if _ess_cf:
                    all_essential_incidence.append(substep.essential_incidence_history)
                if getattr(substep, "layer_attribution", False):
                    all_layer_attribution.append(substep.layer_attribution_history)

            if num_iterations > 1 and all_age_proportions:
                all_runs_np = np.array(all_age_proportions)
                avg_age_proportions = np.mean(all_runs_np, axis=0)
                age_path = f'{output_dir}/counterfactual_data{cf_type}_age_proportions.csv'
                cols = [
                    "t",
                    "child_susceptible", "child_exposed", "child_infected", "child_recovered", "child_dead",
                    "adult_susceptible", "adult_exposed", "adult_infected", "adult_recovered", "adult_dead",
                    "elderly_susceptible", "elderly_exposed", "elderly_infected", "elderly_recovered", "elderly_dead"
                ]
                avg_df = pd.DataFrame(avg_age_proportions, columns=cols)
                avg_df["t"] = avg_df["t"].astype(int)
                avg_df.to_csv(age_path, index=False)

            if all_age_incidence:
                inc_np = np.array(all_age_incidence)              # (iters, T, 4)
                if inc_np.ndim != 3 or inc_np.shape[2] != 4:
                    raise RuntimeError(
                        f"age_incidence has unexpected shape {inc_np.shape} "
                        f"(cf={cf_type}, pop={population})")
                avg_inc = np.mean(inc_np, axis=0)                 # (T, 4)
                inc_df = pd.DataFrame(avg_inc, columns=["t", "child_new", "adult_new", "elderly_new"])
                inc_df["t"] = inc_df["t"].astype(int)
                inc_df.to_csv(f'{output_dir}/counterfactual_data{cf_type}_age_incidence.csv', index=False)

            if all_layer_attribution:
                la_np = np.array(all_layer_attribution)              # (iters, T, 10)
                if la_np.ndim != 3 or la_np.shape[2] != 10:
                    raise RuntimeError(
                        f"layer_attribution has unexpected shape {la_np.shape} "
                        f"(cf={cf_type}, pop={population})")
                la_df = pd.DataFrame(np.mean(la_np, axis=0), columns=[
                    "t", "hh_new", "school_new", "occ_new", "comm_new",
                    "hh_force", "school_force", "occ_force", "comm_force", "new_total"])
                la_df["t"] = la_df["t"].astype(int)
                la_df.to_csv(f'{output_dir}/counterfactual_data{cf_type}_layer_attribution.csv', index=False)

            if all_essential_incidence:
                einc_np = np.array(all_essential_incidence)
                if einc_np.ndim != 3 or einc_np.shape[2] != 4:
                    raise RuntimeError(
                        f"essential_incidence has unexpected shape {einc_np.shape} "
                        f"(cf={cf_type}, pop={population})")
                avg_einc = np.mean(einc_np, axis=0)
                einc_df = pd.DataFrame(avg_einc, columns=["t", "essential_new", "nonessential_new", "other_new"])
                einc_df["t"] = einc_df["t"].astype(int)
                einc_df.to_csv(f'{output_dir}/counterfactual_data{cf_type}_essential_incidence.csv', index=False)

            if cf_type == 7:
                write_factual_run_csvs(sim, output_dir, param_tensor, case_numbers)

        return

    opt = optim.Adam(learn_model.parameters(), lr=0.0005)
    loss_array = np.array([])
    # EPOCHS is a smoke-test override only; production runs leave it unset.
    epochs = int(os.environ.get("EPOCHS", "251") or 251)

    # --- parameter snapshot policy ------------------------------------------
    # The ST-Bernoulli forward sim makes the loss swing ~2x epoch-to-epoch, so
    # the last epoch's params can be an unrepresentative high-loss draw
    # (see COVID-ABM/Discussion/Regularization_discussion.md sec 4.4). With
    # PARAM_SNAPSHOT_K > 1 the saved calibrated_params.txt is the elementwise
    # mean of the last K epochs' parameter vectors, which averages out that
    # jitter. Default 1 -> historical behaviour (last epoch only).
    PARAM_SNAPSHOT_K = max(1, int(os.environ.get("PARAM_SNAPSHOT_K", "1") or 1))
    _param_history = deque(maxlen=PARAM_SNAPSHOT_K)
    _coef_history = deque(maxlen=PARAM_SNAPSHOT_K)

    CURRICULUM_PHASE_1_EPOCHS = 100

    # --- Temporal smoothness penalty on the weekly R sequence -----------------
    # MAP interpretation: a Gaussian random-walk prior on log R_w. Adding the
    # negative log-prior to the data loss ties poorly-identified weeks (e.g. the
    # near-zero-case tail, where SSE gives almost no gradient) to their
    # neighbours, so the calibrated R(t) stays transportable to counterfactuals
    # that perturb the susceptible pool (e.g. CF11 "no vaccines").
    #   penalty = R_SMOOTH_LAMBDA * sum_w ( Delta^order log R_w )^2
    # order 1 -> shrink toward locally-constant R (random walk / RW1)
    # order 2 -> shrink toward locally-linear log R (RW2, smoother trends)
    # R_SMOOTH_ENDPOINT_LAMBDA adds a first-difference anchor on the two boundary
    # weeks:  + lambda_end * [ (dlogR_2)^2 + (dlogR_W)^2 ] . For order>=2 the
    # curvature term alone lets w1 / wW extrapolate the local slope (w1 overshoot,
    # wW float); the anchor pins them to their neighbour. Default 0 (off).
    # Ref: Eilers & Marx 1996 (difference penalties); Abbott et al. 2020 /
    # Flaxman et al. 2020 (random-walk / GP priors on Rt). See
    # COVID-ABM/Discussion/Regularization_discussion.md.
    # Disabled by default (lambda 0) so existing calibrations are unchanged.
    R_SMOOTH_LAMBDA = float(os.environ.get("R_SMOOTH_LAMBDA", "0") or 0.0)
    R_SMOOTH_ORDER = int(os.environ.get("R_SMOOTH_ORDER", "1") or 1)
    R_SMOOTH_ENDPOINT_LAMBDA = float(
        os.environ.get("R_SMOOTH_ENDPOINT_LAMBDA", "0") or 0.0
    )
    if R_SMOOTH_LAMBDA > 0:
        print(f"[INFO] R smoothness penalty ON: lambda={R_SMOOTH_LAMBDA}, "
              f"order={R_SMOOTH_ORDER} (RW{R_SMOOTH_ORDER} on log R), "
              f"endpoint_lambda={R_SMOOTH_ENDPOINT_LAMBDA}", flush=True)

    for epoch in range(epochs):
        sim.config['simulation_metadata']['is_final_epoch'] = (epoch == epochs - 1)
        torch.autograd.set_detect_anomaly(True)
        opt.zero_grad()

        if with_vacc:
            if epoch < CURRICULUM_PHASE_1_EPOCHS:
                sim.config['simulation_metadata']['WITH_VACC'] = False
                cutoff_step = None
            else:
                sim.config['simulation_metadata']['WITH_VACC'] = True
                cutoff_step = None
        else:
            cutoff_step = None

        runner.state = deep_clone_state(initial_state)
        runner.state_trajectory = []
        
        debug_tensor, spline_coef = learn_model(context_tensor)
        debug_tensor = debug_tensor[:, None]

        map_and_replace_tensor(learnable_params[1][0])(runner, False, debug_tensor[:num_weeks], mode_calibrate=True)
        map_and_replace_tensor(learnable_params[3][0])(runner, False, debug_tensor[-2], mode_calibrate=True)
        map_and_replace_tensor(learnable_params[4][0])(runner, False, debug_tensor[-1], mode_calibrate=True)

        loss = execute(sim, runner, case_numbers, epoch, epochs, num_steps, loss_cutoff_step=cutoff_step)

        data_loss_np = float(loss.detach().cpu())
        r_smooth_pen_np = 0.0
        if R_SPLINE:
            d2c = D2_t @ spline_coef
            r_smooth_pen = R_SPLINE_LAMBDA * d2c.pow(2).sum()
            if R_SPLINE_END_LAMBDA > 0 and spline_coef.numel() >= 2:
                d1 = spline_coef[1:] - spline_coef[:-1]
                r_smooth_pen = r_smooth_pen + R_SPLINE_END_LAMBDA * (
                    d1[0] ** 2 + d1[-1] ** 2
                )
            if not torch.isfinite(r_smooth_pen):
                raise RuntimeError(
                    f"spline penalty non-finite at epoch {epoch}: "
                    f"pen={float(r_smooth_pen)}, coef range="
                    f"[{float(spline_coef.min()):.4f},"
                    f"{float(spline_coef.max()):.4f}]")
            loss = loss + r_smooth_pen
            r_smooth_pen_np = float(r_smooth_pen.detach().cpu())
        elif R_SMOOTH_LAMBDA > 0:
            log_R = torch.log(debug_tensor[:num_weeks, 0].clamp_min(1e-6))
            diff = log_R
            for _ in range(R_SMOOTH_ORDER):
                diff = diff[1:] - diff[:-1]
            r_smooth_pen = R_SMOOTH_LAMBDA * (diff ** 2).sum()
            if R_SMOOTH_ENDPOINT_LAMBDA > 0 and log_R.numel() >= 2:
                d1 = log_R[1:] - log_R[:-1]
                r_smooth_pen = r_smooth_pen + R_SMOOTH_ENDPOINT_LAMBDA * (
                    d1[0] ** 2 + d1[-1] ** 2
                )
            loss = loss + r_smooth_pen
            r_smooth_pen_np = float(r_smooth_pen.detach().cpu())

        loss.backward()
        opt.step()

        if R_SPLINE and (epoch % 25 == 0 or epoch == epochs - 1):
            with torch.no_grad():
                R_w_np = debug_tensor[:num_weeks, 0].detach().cpu().numpy()
                c_np = spline_coef.detach().cpu().numpy()
            pen_share = r_smooth_pen_np / (data_loss_np + r_smooth_pen_np + 1e-12)
            print(f"[ep {epoch:4d}] data={data_loss_np:9.3f} "
                  f"pen={r_smooth_pen_np:8.4f} share={pen_share:.4f} "
                  f"EDF={spline_edf:5.2f} "
                  f"R=[{R_w_np.min():.3f},{R_w_np.max():.3f}] "
                  f"c=[{c_np.min():.3f},{c_np.max():.3f}] "
                  f"i0={float(debug_tensor[-2, 0]):.6f} "
                  f"k={float(debug_tensor[-1, 0]):.4f}", flush=True)
            if not np.isfinite(R_w_np).all():
                raise RuntimeError(
                    f"non-finite weekly R at epoch {epoch}: {R_w_np}")
            if (R_w_np <= 0).any():
                raise RuntimeError(
                    f"non-positive weekly R at epoch {epoch}: "
                    f"{R_w_np[R_w_np <= 0]} (impossible with the log "
                    f"parameterization -- check the basis)")
            _lo, _hi = np.log(R_SPLINE_R_MIN), np.log(R_SPLINE_R_MAX)
            sat = (c_np - _lo) / (_hi - _lo)
            _bad = np.where((sat < 0.02) | (sat > 0.98))[0]
            if _bad.size:
                raise RuntimeError(
                    f"coefficient box saturated at epoch {epoch}, indices "
                    f"{_bad.tolist()} (values {c_np[_bad].tolist()}). Those "
                    f"coefficients have near-zero gradient. Widen "
                    f"R_SPLINE_R_MIN / R_MAX.")
        elif R_SMOOTH_LAMBDA > 0 and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"[epoch {epoch}] data_loss={data_loss_np:.2f}  "
                  f"R_smooth_pen={r_smooth_pen_np:.2f}  total={data_loss_np + r_smooth_pen_np:.2f}",
                  flush=True)

        loss_np = loss.detach().cpu().numpy()
        loss_array = np.append(loss_array, loss_np)

        _param_history.append(debug_tensor.detach().cpu().numpy())
        if R_SPLINE:
            _coef_history.append(spline_coef.detach().cpu().numpy())

        if epoch == epochs - 1:
            substep = runner.initializer.transition_function['0'].new_transmission
            output_dir = f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}_{use_7day_avg}_metro_{metro_phase}'
            os.makedirs(output_dir, exist_ok=True)  # not yet created when EPOCHS <= curriculum length
            path = f'{output_dir}/training_proportions.csv'
            substep.save_proportions_to_disk(path)
            if PARAM_SNAPSHOT_K > 1 and len(_param_history) > 1:
                params_to_save = np.mean(np.stack(list(_param_history), axis=0), axis=0)
                print(f"[INFO] saving calibrated_params.txt as mean of last "
                      f"{len(_param_history)} epochs (PARAM_SNAPSHOT_K)", flush=True)
            else:
                params_to_save = debug_tensor.detach().cpu().numpy()
            np.savetxt(os.path.join(output_dir, "calibrated_params.txt"), params_to_save, fmt="%.6f")

            if R_SPLINE:
                # Sidecars: the 28-row calibrated_params.txt contract above is
                # untouched (six downstream consumers index it positionally),
                # so the spline state is recorded separately.
                coef_to_save = (np.mean(np.stack(list(_coef_history), axis=0), axis=0)
                                if PARAM_SNAPSHOT_K > 1 and len(_coef_history) > 1
                                else spline_coef.detach().cpu().numpy())
                np.savetxt(os.path.join(output_dir, "spline_coef.txt"),
                           coef_to_save, fmt="%.8f")
                _log_r_daily = B_np @ coef_to_save.astype(np.float64)
                # r_daily.txt keeps one row per observation day (182) in every
                # variant so R_curve.png stays directly comparable across runs
                # (session_summary.md 15.3). Under R_SPLINE_SCORED_ONLY the
                # basis only spans the 168 scored days, so the unscored tail is
                # written as the flat continuation the weekly values already
                # encode -- not extrapolated.
                if len(_log_r_daily) < _obs_days:
                    _log_r_daily = np.concatenate([
                        _log_r_daily,
                        np.repeat(_log_r_daily[-1], _obs_days - len(_log_r_daily)),
                    ])
                np.savetxt(os.path.join(output_dir, "r_daily.txt"),
                           np.exp(_log_r_daily), fmt="%.8f")
                with open(os.path.join(output_dir, "spline_meta.json"), "w") as _f:
                    json.dump({
                        "J": R_SPLINE_J,
                        "degree": R_SPLINE_DEGREE,
                        "lambda": R_SPLINE_LAMBDA,
                        "endpoint_lambda": R_SPLINE_END_LAMBDA,
                        "edf": spline_edf,
                        "log_r_min": float(np.log(R_SPLINE_R_MIN)),
                        "log_r_max": float(np.log(R_SPLINE_R_MAX)),
                        "n_days": int(num_steps),
                        "num_weeks": int(num_weeks),
                        "scored_only": bool(_scored_only),
                        "spline_days": int(_spline_days),
                        "spline_weeks": int(_spline_weeks),
                        "eval": "weekly_midpoint",
                        "cond_B_w": float(np.linalg.cond(B_w_np)),
                        "init_r": (float(os.environ["R_SPLINE_INIT_R"])
                                   if os.environ.get("R_SPLINE_INIT_R", "").strip()
                                   else None),
                        "param_snapshot_k": PARAM_SNAPSHOT_K,
                        "epochs": epochs,
                        "seed": sim.config['simulation_metadata'].get('random_seed'),
                        "loss_mode": os.environ.get("LOSS_MODE", "sse"),
                    }, _f, indent=2)
            
            if metro_phase == 1:
                results_dir = f'results/{population}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_metro_1'
                os.makedirs(results_dir, exist_ok=True)
                model_path = os.path.join(results_dir, "calibrated_model.pt")
                print(f"Saving Phase 1 weights to {model_path}...")
                state_to_save = learn_model._orig_mod.state_dict() if hasattr(learn_model, '_orig_mod') else learn_model.state_dict()
                torch.save(state_to_save, model_path)

    iters = np.arange(1, epochs + 1)
    data = np.column_stack((iters, loss_array))

    output_dir = f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}_{use_7day_avg}_metro_{metro_phase}'
    with open(os.path.join(output_dir, "training_loss.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration', 'Task Loss'])
        writer.writerows(data)
