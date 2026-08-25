# Project Next Steps & Roadmap

This document outlines the technical roadmap and next steps for the Epidemic Differentiable ABM repository, compiled from collaborator meetings with epidemiologists and internal team synthesis (Preetom's notes).

---

## Pillar 1: Epidemiological & Immune Dynamics

### 1.1 Waning Immunity Implementation ($R \rightarrow S$)
* **Context**: Epidemiological data indicates immunity wanes over time (approx. 6-month half-life). Without waning immunity, the susceptible pool becomes artificially depleted over long simulation horizons.
* **Goal**: Implement exponential waning of immunity following the Kit McLean model formulation so recovered individuals transition back to susceptible ($R \rightarrow S$).
* **Implementation Files**:
  * [covid_abm/substeps/new_transmission/transition.py](file:///home/facundoy/research/epi-diff-abm-dev/covid_abm/substeps/new_transmission/transition.py):
    * Update `NewTransmission` to evaluate agent `recovered_time`.
    * Apply exponential decay probability $P_{wane}(t) = 1 - e^{-\lambda_{wane} \cdot (t - t_{rec})}$ to switch `disease_stage` from `RECOVERED_VAR` (3) back to `SUSCEPTIBLE_VAR` (0).
  * [covid_abm/yamls/config.yaml](file:///home/facundoy/research/epi-diff-abm-dev/covid_abm/yamls/config.yaml):
    * Add metadata configuration fields: `RECOVERED_TO_SUSCEPTIBLE_TIME: 180` (6 months) and `WANING_RATE: 0.0055`.

### 1.2 Initial Immunity & Learnable Initial Recovered Proportion ($R_0$)
* **Context**: At simulation start ($t=0$), assuming 100% of non-infected agents are susceptible underestimates prior population exposure.
* **Goal**: Start simulation with a non-zero fraction of agents in the `RECOVERED` stage at $t=0$, and make this initial recovered percentage a learnable model parameter.
* **Implementation Files**:
  * [abm_nets.py](file:///home/facundoy/research/epi-diff-abm-dev/abm_nets.py):
    * Extend `LearnableParams` class to include `initial_recovered_ratio` as an optimizable PyTorch parameter.
  * [networks/custom_population.py](file:///home/facundoy/research/epi-diff-abm-dev/networks/custom_population.py) & [covid_abm/yamls/config.yaml](file:///home/facundoy/research/epi-diff-abm-dev/covid_abm/yamls/config.yaml):
    * Update population generation to assign initial disease stages using `initial_recovered_ratio`.

---

## Pillar 2: Policy Interventions & Counterfactual Scenarios

### 2.1 Consistent Policy Implementation Probability (Seed Maintenance)
* **Context**: Comparing factual vs. counterfactual runs requires isolating policy changes from random agent compliance noise.
* **Goal**: Maintain strict PRNG seeds and compliance probability vectors across policy flips ($0 \leftrightarrow 1$).
* **Implementation Files**:
  * [abm_nets.py](file:///home/facundoy/research/epi-diff-abm-dev/abm_nets.py):
    * Reset PyTorch/NumPy RNG seeds to identical values prior to executing each counterfactual variant loop.
    * Enforce constant closure percentage/compliance rates across all runs and intervention types.

### 2.2 Expanding Counterfactual Scenario Set & Real-World Policy Alignment
* **Context**: Binary policy flipping ($0 \leftrightarrow 1$) is abstract. Epidemiologists requested actionable policy scenarios (varying timing/duration, specific holiday extensions, workplace open / school factual).
* **Goal**: Extend `cf_types_to_run` to include:
  * **Work Open / School Factual** (Workplace unrestricted, school historical).
  * **All Closed + All Vaccines** (Maximum restriction scenario + full vaccine coverage).
  * **Full Permutation Coverage** (Complete matrix of school/workplace options).
  * **Timing & Duration Variations** (e.g., 2-week early closure, 4-week delayed reopening).
* **Implementation Files**:
  * [abm_nets.py](file:///home/facundoy/research/epi-diff-abm-dev/abm_nets.py):
    * Add new counterfactual type IDs (11–16) in `eval_net` and expand policy modification logic.
  * [scripts/process_counties.py](file:///home/facundoy/research/epi-diff-abm-dev/scripts/process_counties.py):
    * Add support for generating time-shifted intervention matrices (`intervention.csv`).

---

## Pillar 3: Vaccination Dynamics

### 3.1 Dual-Tier Vaccination Modeling
* **Context**: Current models treat vaccination homogeneously. Epidemiologists recommended distinguishing widely available baseline vaccination from periodic booster campaigns.
* **Goal**: Incorporate empirical CDC/Delphi vaccination uptake curves and model:
  1. Widely available continuous vaccination.
  2. Infrequent periodic booster campaigns.
* **Implementation Files**:
  * [scripts/delphi_api.py](file:///home/facundoy/research/epi-diff-abm-dev/scripts/delphi_api.py):
    * Pull historical county-level vaccination rates from Delphi COVIDcast API.
  * [covid_abm/substeps/new_transmission/transition.py](file:///home/facundoy/research/epi-diff-abm-dev/covid_abm/substeps/new_transmission/transition.py):
    * Apply time-dependent susceptibility multipliers based on agent vaccination status and time since last dose.

---

## Pillar 4: Calibration, Visualization, & Unidentifiability Investigation

### 4.1 Early Scaling Factor & Initial Infection Rate Optimization
* **Context**: Calibration currently relies on early-week scaling factors to compensate for initial infection rates ($I_0$).
* **Goal**: Adjust the early scaling factor and increase $I_0$ initialization so learnable transmission parameters ($\beta, R$) reflect true contact dynamics rather than compensating for early miscalibration.
* **Implementation Files**:
  * [abm_nets.py](file:///home/facundoy/research/epi-diff-abm-dev/abm_nets.py):
    * Re-tune early scaling multiplier in loss function calculation (`eval_net`).

### 4.2 Parameter Unidentifiability Investigation
* **Context**: Unidentifiability occurs when multiple parameter combinations (e.g., underreporting factor $k$ vs. transmission rate $R$) produce identical loss values.
* **Goal**: Perform systematic sensitivity analysis and profile likelihood checks across $(k, R, I_0)$ parameter space to identify and resolve unidentifiability.
* **Implementation Files**:
  * [abm_nets.py](file:///home/facundoy/research/epi-diff-abm-dev/abm_nets.py):
    * Add multi-start parameter initialization and log parameter Hessian / Jacobian condition numbers.

### 4.3 Multi-Run Ensemble Averaging & Ribbon Visualization
* **Context**: Single trajectory lines do not reflect stochastic variance.
* **Goal**:
  1. Compute ensemble averages across stochastically seeded calibration runs.
  2. Add shaded 95% confidence ribbons/bands to output trajectory plots.
  3. Overlay ground-truth empirical data lines on all counterfactual comparison plots.
* **Implementation Files**:
  * [collect_simulation_graphs.py](file:///home/facundoy/research/epi-diff-abm-dev/collect_simulation_graphs.py):
    * Update plot generation to calculate mean and 95% confidence intervals (`plt.fill_between`).
    * Plot empirical ground-truth case/death time series as a distinct reference line.
