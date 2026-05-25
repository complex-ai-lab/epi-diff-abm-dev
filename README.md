# ABM-Benchmark for COVID-19 Modeling

This repository provides the implementation of an Agent-Based Model (ABM) benchmark for COVID-19 transmission and intervention modeling.

## How to run
To run the simulations, follow the instructions below.

### Setup environment
Use `environment.yml` to create the corresponding conda environment:
```bash
conda env create -f environment.yml
conda activate covid_env
```

### Step 1: Data preparation
First, initialize the submodules to include necessary dependencies:
```bash
git submodule update --init --recursive
```

The final pre-processed dataset can also be found in the following link: [LINK_TO_DATASET_PENDING]

**Preparation Order:**
The data preparation process consists of two main stages:
1.  **`scripts/`**: Run the scripts in the `scripts/` directory first to process raw county-level data into simulation-ready formats.
2.  **`networks/`**: After processing the data, run the scripts in the `networks/` directory to generate the mobility networks.

Both the `scripts/` and `networks/` directories contain their own respective `README.md` files with detailed instructions.

### Step 2: Main Experiment
The core simulation and calibration logic is managed by `abm_nets.py`, which is executed through `main.py`.

#### Configuration Requirements
Before running an experiment, ensure the following configurations are correctly set:

**1. `abm_nets.py` Modifications:**
*   **Population Import**: At the top of the file, update the population import statement (e.g., `from populations import pop26007`).
*   **Population Loader**: In the `eval_net` function, update the `Executor` initialization to use the correct population module (e.g., `sim = Executor(covid_abm, pop_loader=LoadPopulation(pop26007))`).
*   **Averaging Range**: If generating counterfactuals, update `cf_types_to_run` (e.g., `range(7, 11)`) to specify which intervention types to execute.

**2. `covid_abm/yamls/config.yaml` Modifications:**
*   **`POPULATION`**: Specify the FIPS code of the target county.
*   **`GENERATING_COUNTERFACTUAL`**: Set to `true` to run counterfactual scenarios (requires calibrated parameters), or `false` to perform training/factual calibration.

#### Counterfactual Types
The simulation supports 10 distinct intervention scenarios (Counterfactual Types), defined by combinations of School and Occupational (Occ) restrictions:

| Type | School | Occ | Description |
| :--- | :--- | :--- | :--- |
| **1** | 0 | 0 | Both fixed to 0 (No restrictions) |
| **2** | 1 | 0 | School fixed to 1, Occ fixed to 0 |
| **3** | 0 | 1 | School fixed to 0, Occ fixed to 1 |
| **4** | 1 | 1 | Both fixed to 1 (Full restrictions) |
| **5** | 0 | Factual | School fixed to 0, Occ remains Factual |
| **6** | 0 | Counterfactual | School fixed to 0, Occ is Flipped |
| **7** | Factual | Factual | **Baseline Factual scenario** |
| **8** | Counterfactual | Factual | School is Flipped, Occ remains Factual |
| **9** | Factual | Counterfactual | School remains Factual, Occ is Flipped |
| **10** | Counterfactual | Counterfactual | **Both School and Occ are Flipped** |

*Note: "Flipped" (Counterfactual) means the intervention value is toggled (0 becomes 1, 1 becomes 0) relative to the historical data.*

To run the experiment:
```bash
python main.py
```
