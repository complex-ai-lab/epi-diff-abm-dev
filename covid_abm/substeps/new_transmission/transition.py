import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import re
import math
import pandas as pd
import numpy as np
import os

from agent_torch.core.substep import SubstepTransitionMessagePassing
from agent_torch.core.helpers import get_by_path
from agent_torch.core.helpers import set_by_path
from agent_torch.core.distributions import StraightThroughBernoulli

class NewTransmission(SubstepTransitionMessagePassing):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]
        self.RECOVERED_VAR = self.config["simulation_metadata"]["RECOVERED_VAR"]
        self.INFECTED_VAR = self.config["simulation_metadata"]["INFECTED_VAR"]
        self.MORTALITY_VAR = self.config["simulation_metadata"]["MORTALITY_VAR"]

        self.num_timesteps = self.config["simulation_metadata"]["num_steps_per_episode"]
        self.num_weeks = self.config["simulation_metadata"]["NUM_WEEKS"]

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.data_dir = os.path.join(project_root, 'data')

        self.networks = self._preload_all_networks()
        self.household_net = self._load_single_net(f"{self.data_dir}/networks/covid_output_causal/{self.config['simulation_metadata']['POPULATION']}/mobility_networks/HOUSEHOLD_NETWORK.pkl")
        self.school_net = self._load_single_net(f"{self.data_dir}/networks/covid_output_causal/{self.config['simulation_metadata']['POPULATION']}/mobility_networks/SCHOOL_NETWORK.pkl")
        self.proportion_history = []
        self.age_proportion_history = []

        self.STAGE_UPDATE_VAR = 1
        self.INFINITY_TIME = self.config["simulation_metadata"]["INFINITY_TIME"]
        self.EXPOSED_TO_INFECTED_TIME = self.config["simulation_metadata"][
            "EXPOSED_TO_INFECTED_TIME"
        ]
        self.INFECTED_TO_RECOVERED_TIME = self.config["simulation_metadata"][
            "INFECTED_TO_RECOVERED_TIME"
        ]

        self.mode = self.config["simulation_metadata"]["EXECUTION_MODE"]
        self.st_bernoulli = StraightThroughBernoulli.apply

        self.calibration_mode = self.config['simulation_metadata']['calibration']

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
        edge_network_numbers = edge_attr[0, :]

        I_bar = torch.gather(x_i[:, 4], 0, edge_network_numbers.long()).view(-1)

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

        self.proportion_history.append([t] + proportions.detach().cpu().tolist())

    def save_proportions_to_disk(self, output_path):
        if self.proportion_history:
            df = pd.DataFrame(self.proportion_history, 
                            columns=["t", "susceptible", "exposed", "infected", "recovered", "dead"])
            df.to_csv(output_path, index=False)

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

            if os.path.exists(output_path):
                existing_df = pd.read_csv(output_path)

                if "t" in existing_df.columns:
                    final_df = pd.merge(existing_df, new_df, on="t", how="outer")
                else:
                    final_df = pd.concat([existing_df, new_df.drop(columns=["t"])], axis=1)
            else:
                final_df = new_df

            final_df.to_csv(output_path, index=False)

    def modify_initial_infected(self, current_stages, proportion, tau=0.5, hard=True):
        device = current_stages.device
        N = int(self.config['simulation_metadata']['num_agents'])

        new_stages = current_stages.clone()
        new_stages[:, 0] = 0.0

        logits = torch.zeros(N, 2, device=device)
        logits[:, 1] = torch.log(torch.tensor(proportion / (1 - proportion), device=device))

        samples = F.gumbel_softmax(logits, tau=tau, hard=hard)
        infected_mask = samples[:, 1]

        new_stages[:, 0] = 2.0 * infected_mask

        return new_stages

    def get_mean_agent_interactions(self, agents_ages):
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
            nets['occ'].append(self._load_single_net(f"{self.data_dir}/networks/covid_output_causal/{population}/mobility_networks/occnets/{t}.pkl"))
            nets['rand'].append(self._load_single_net(f"{self.data_dir}/networks/covid_output_causal/{population}/mobility_networks/randnets/{t}.pkl"))
        return nets

    def apply_intervention_fast(self, intervention, edges):
        if edges.size(0) == 0: return edges

        if intervention == 0:
            sample = torch.clamp(torch.normal(0.5, 0.15, (1,)), 0, 0.25).to(self.device)
        else:
            sample = torch.clamp(torch.normal(0.5, 0.15, (1,)), 0.75, 1).to(self.device)

        keep_frac = (1.0 - sample).item()
        k = int(edges.size(0) * keep_frac)

        idx = torch.randperm(edges.size(0), device=self.device)[:k]
        return edges[idx]
    
    def apply_vaccines(self, current_stages, num_vaccines, tau=0.1):
        device = current_stages.device
        N = current_stages.shape[0]
        
        is_susceptible = self.soft_eq(current_stages, self.SUSCEPTIBLE_VAR, temperature=tau)
        logits = torch.log(is_susceptible + 1e-10).view(N, 1)
        
        gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y = logits + gumbels
        
        num_vax_int = int(num_vaccines.item()) if torch.is_tensor(num_vaccines) else int(num_vaccines)
        num_vax_int = min(num_vax_int, int(is_susceptible.sum().item()))
        
        if num_vax_int <= 0:
            return current_stages

        _, indices = torch.topk(y.view(-1), num_vax_int)
        
        hard_mask = torch.zeros(N, 1, device=device)
        hard_mask[indices] = 1.0
        
        soft_probs = torch.sigmoid(logits) 
        vax_mask = (hard_mask - soft_probs).detach() + soft_probs
        
        stage_delta = (self.RECOVERED_VAR - self.SUSCEPTIBLE_VAR) * vax_mask
        updated_stages = current_stages + stage_delta
        
        return updated_stages

    def get_age_stage_proportions(self, t, current_stages, agents_ages):
            ADULT_LOWER_INDEX, ADULT_UPPER_INDEX = 1, 4

            child_agents = (agents_ages < ADULT_LOWER_INDEX).view(-1)
            adult_agents = torch.logical_and(
                agents_ages >= ADULT_LOWER_INDEX, agents_ages <= ADULT_UPPER_INDEX
            ).view(-1)
            elderly_agents = (agents_ages > ADULT_UPPER_INDEX).view(-1)

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

        school_intervention = curr_intervention.iloc[0]['school_intervention']
        occ_intervention = curr_intervention.iloc[0]['occ_intervention']
        num_vaccines = curr_intervention.iloc[0]['vaccines']

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
                10: ("CF", "CF")
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

        school_net = self.apply_intervention_fast(school_intervention, self.school_net)
        occ_net = self.apply_intervention_fast(occ_intervention, self.networks['occ'][t])

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
            commute_file = f"{self.data_dir}/networks/covid_output_causal/{population}/mobility_networks/occnets/{population}_residence_commute_data.txt"
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

        if with_vacc:
            current_stages = self.apply_vaccines(current_stages, num_vaccines)

        daily_deaths = self.update_number_of_dead(daily_deaths, current_stages, agents_next_stage_times, t, newly_exposed_today)

        updated_stages = self.update_stages(t, current_stages, agents_next_stage_times, newly_exposed_today)
        updated_next_stage_times = self.update_transition_times(
            t, agents_next_stage_times, newly_exposed_today, current_stages
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