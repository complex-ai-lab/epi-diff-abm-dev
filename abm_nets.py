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
import pandas as pd


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
    7:  "factual school, factual work",
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
}


class LearnableParams(nn.Module):
    def __init__(self, num_params, context_dim=3, device=DEVICE):
        super().__init__()
        self.device = device
        self.num_params = num_params
        
        self.fc1 = nn.Linear(context_dim, 64).to(self.device)
        self.fc2 = nn.Linear(64, 32).to(self.device)
        self.fc3 = nn.Linear(32, self.num_params + 2).to(self.device)
        self.ReLU = nn.ReLU()
        self.learnable_params = nn.Parameter(torch.rand(num_params + 2, device=self.device))
        
        # Upper bound on the weekly R multipliers. Historically 5.0; env-tunable
        # so the non-identified tail can be capped (see
        # COVID-ABM/Discussion/Regularization_discussion.md 5.7). The two
        # trailing params (infection-rate scale, k) keep their own bounds.
        r_max = float(os.environ.get("R_MAX", "5") or 5.0)
        min_values_list = [0] * num_params + [0.0] + [0.2]
        max_values_list = [r_max] * num_params + [0.0005] + [1]

        self.min_values = torch.tensor(min_values_list, device=self.device)
        self.max_values = torch.tensor(max_values_list, device=self.device)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.to(self.device)
        out = self.ReLU(self.fc1(x))
        out = self.ReLU(self.fc2(out))
        out = self.fc3(out)
        out = self.min_values + (self.max_values -
                                 self.min_values) * self.sigmoid(out)
        return out

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

    start = n_steps - weeks_to_eval * 7
    labels = labels[start:]
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
    learn_model = LearnableParams(num_weeks, context_dim=3, device=DEVICE)
    
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

        if not os.path.exists(param_file):
            print(f"\n[ERROR] Calibrated parameter file not found at: '{param_file}'")
            print(f"[ERROR] Cannot run counterfactual simulations without calibrated parameters.")
            print(f"[ERROR] Please run model calibration first for county {population} (using run_all_sims.sh or main.py).\n")
            sys.exit(1)

        try:
            param_array = np.loadtxt(param_file)
        except Exception as e:
            print(f"\n[ERROR] Failed to load calibrated parameters from '{param_file}': {e}\n")
            sys.exit(1)

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
                if _ess_cf:
                    substep.save_essential_incidence_to_disk(
                        f'{output_dir}/counterfactual_data{cf_type}_essential_incidence_raw.csv', _it)

                all_age_proportions.append(substep.age_proportion_history)
                all_age_incidence.append(substep.age_incidence_history)
                if _ess_cf:
                    all_essential_incidence.append(substep.essential_incidence_history)

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

        return

    opt = optim.Adam(learn_model.parameters(), lr=0.0005)
    loss_array = np.array([])
    epochs = 251

    # --- parameter snapshot policy ------------------------------------------
    # The ST-Bernoulli forward sim makes the loss swing ~2x epoch-to-epoch, so
    # the last epoch's params can be an unrepresentative high-loss draw
    # (see COVID-ABM/Discussion/Regularization_discussion.md sec 4.4). With
    # PARAM_SNAPSHOT_K > 1 the saved calibrated_params.txt is the elementwise
    # mean of the last K epochs' parameter vectors, which averages out that
    # jitter. Default 1 -> historical behaviour (last epoch only).
    PARAM_SNAPSHOT_K = max(1, int(os.environ.get("PARAM_SNAPSHOT_K", "1") or 1))
    _param_history = deque(maxlen=PARAM_SNAPSHOT_K)

    CURRICULUM_PHASE_1_EPOCHS = 100
    VACCINE_ROLLOUT_DAY = 14

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
        
        debug_tensor = learn_model(context_tensor)
        debug_tensor = debug_tensor[:, None]
        
        map_and_replace_tensor(learnable_params[1][0])(runner, False, debug_tensor[:num_weeks], mode_calibrate=True)
        map_and_replace_tensor(learnable_params[3][0])(runner, False, debug_tensor[-2], mode_calibrate=True)
        map_and_replace_tensor(learnable_params[4][0])(runner, False, debug_tensor[-1], mode_calibrate=True)

        loss = execute(sim, runner, case_numbers, epoch, epochs, num_steps, loss_cutoff_step=cutoff_step)

        data_loss_np = float(loss.detach().cpu())
        r_smooth_pen_np = 0.0
        if R_SMOOTH_LAMBDA > 0:
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

        if R_SMOOTH_LAMBDA > 0 and (epoch % 50 == 0 or epoch == epochs - 1):
            print(f"[epoch {epoch}] data_loss={data_loss_np:.2f}  "
                  f"R_smooth_pen={r_smooth_pen_np:.2f}  total={data_loss_np + r_smooth_pen_np:.2f}",
                  flush=True)

        loss_np = loss.detach().cpu().numpy()
        loss_array = np.append(loss_array, loss_np)

        _param_history.append(debug_tensor.detach().cpu().numpy())

        if epoch == epochs - 1:
            substep = runner.initializer.transition_function['0'].new_transmission
            output_dir = f'result_graphs/{population}/{date}/{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}_{use_7day_avg}_metro_{metro_phase}'
            path = f'{output_dir}/training_proportions.csv'
            substep.save_proportions_to_disk(path)
            if PARAM_SNAPSHOT_K > 1 and len(_param_history) > 1:
                params_to_save = np.mean(np.stack(list(_param_history), axis=0), axis=0)
                print(f"[INFO] saving calibrated_params.txt as mean of last "
                      f"{len(_param_history)} epochs (PARAM_SNAPSHOT_K)", flush=True)
            else:
                params_to_save = debug_tensor.detach().cpu().numpy()
            np.savetxt(os.path.join(output_dir, "calibrated_params.txt"), params_to_save, fmt="%.6f")
            
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
