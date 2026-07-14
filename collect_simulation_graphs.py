#!/usr/bin/env python3
"""
collect_simulation_graphs.py

Locates the final simulation_results.png graphs for all 153 counties 
hardcoded in run_all_sims.sh, constructs the correct paths based on 
covid_abm/yamls/config.yaml, and copies them to the 'all_simulation_results' folder.
"""

import os
import shutil
import yaml

# Hardcoded list of all 153 counties from run_all_sims.sh
COUNTIES = [
    '01009', '01031', '01039', '01045', '01049', '01071', '13013', '13071', '13103', '13115',
    '13137', '13153', '17011', '17021', '17037', '17055', '17063', '17073', '17091', '17095',
    '17103', '17115', '17117', '17121', '17141', '17167', '19017', '19027', '19113', '19155',
    '19169', '20035', '20045', '20061', '20079', '20103', '20125', '20149', '20161', '20169',
    '21009', '21029', '21035', '21071', '21073', '21083', '21089', '21093', '21113', '21179',
    '21199', '21209', '21211', '22001', '22005', '22039', '22063', '26023', '26027', '26041',
    '26057', '26059', '26067', '26073', '26123', '27005', '27027', '27041', '27059', '27085',
    '27109', '29021', '29027', '29051', '29101', '30029', '30047', '30049', '30093', '36003',
    '36011', '36013', '36031', '36051', '38059', '38077', '38093', '39005', '39007', '39011',
    '39013', '39015', '39021', '39023', '39031', '39033', '39037', '39039', '39045', '39051',
    '39057', '39059', '39063', '39071', '39077', '39079', '39083', '39087', '40021', '40089',
    '40097', '40121', '40147', '42005', '42013', '42025', '42031', '42051', '42055', '42059',
    '42061', '42075', '42085', '42087', '46011', '46013', '46029', '46035', '46081', '46083',
    '46135', '47003', '47005', '47009', '47017', '48001', '48005', '48013', '48041', '48049',
    '48071', '48091', '48097', '48143', '48147', '48181', '48189', '48213', '51047', '51095',
    '51137', '51149', '51161'
]

def load_config():
    config_path = os.path.join("covid_abm", "yamls", "config.yaml")
    if not os.path.exists(config_path):
        # Fallback to check directly inside covid_abm/ if yamls/ isn't used
        config_path = os.path.join("covid_abm", "config.yaml")
        
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find config.yaml in covid_abm/yamls/ or covid_abm/.")
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    dest_dir = "all_simulation_results"
    os.makedirs(dest_dir, exist_ok=True)

    try:
        config = load_config()
        meta = config.get("simulation_metadata", {})
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return

    # Extract simulation parameter values
    date = meta.get("DATE", "202010-202104")
    initial_rate = meta.get("INITIAL_INFECTION_RATE", 0.0005)
    exposed_to_infected = meta.get("EXPOSED_TO_INFECTED_TIME", 3)
    infected_to_recovered = meta.get("INFECTED_TO_RECOVERED_TIME", 5)
    with_k = meta.get("WITH_K", True)
    with_vacc = meta.get("WITH_VACC", True)
    metro_phase = meta.get("metro_calibration_phase", 0)

    # Format parameter folder name
    param_folder = f"{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_{with_vacc}_metro_{metro_phase}"

    print("--- Simulation Parameters Detected ---")
    print(f"Date:                  {date}")
    print(f"Initial Infection Rate:{initial_rate}")
    print(f"Exposed to Infected:   {exposed_to_infected}")
    print(f"Infected to Recovered: {infected_to_recovered}")
    print(f"With K:                {with_k}")
    print(f"With Vaccine:          {with_vacc}")
    print(f"Metro Calibration Phase:{metro_phase}")
    print(f"Parameter Folder:      {param_folder}")
    print(f"Destination Directory:  {dest_dir}")
    print("---------------------------------------\n")

    copied_count = 0
    missing_count = 0

    for county in COUNTIES:
        # Check standard last epoch folders (500 for 500-epoch runs, 250 for 250-epoch runs)
        epochs_to_check = [500, 499, 250, 200, 150, 100, 50, 0]
        
        # Build list of potential directory options to check in case parameters changed slightly
        # (e.g., WITH_VACC was set to False in some runs)
        potential_param_folders = [
            param_folder,
            # Fallbacks with different WITH_VACC/WITH_K boolean variants
            f"{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_False_metro_{metro_phase}",
            f"{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_{with_k}_True_metro_{metro_phase}",
            f"{initial_rate}_{exposed_to_infected}_{infected_to_recovered}_False_False_metro_{metro_phase}"
        ]
        
        # Remove duplicates while preserving order
        unique_param_folders = []
        for pf in potential_param_folders:
            if pf not in unique_param_folders:
                unique_param_folders.append(pf)

        found = False
        for pf in unique_param_folders:
            for epoch in epochs_to_check:
                src_path = os.path.join(
                    "result_graphs",
                    county,
                    date,
                    pf,
                    str(epoch),
                    "simulation_results.png"
                )

                if os.path.exists(src_path):
                    dest_filename = f"{county}_simulation_results.png"
                    dest_path = os.path.join(dest_dir, dest_filename)
                    shutil.copy2(src_path, dest_path)
                    print(f"[{county}] Copied from epoch {epoch} run folder '{pf}'")
                    copied_count += 1
                    found = True
                    break
            if found:
                break

        if not found:
            # Final attempt: try glob search in case the parameters are completely different
            import glob
            fallback_pattern = os.path.join("result_graphs", county, date, "*", "*", "simulation_results.png")
            matches = glob.glob(fallback_pattern)
            
            if matches:
                # Find the match with the highest epoch number
                highest_epoch = -1
                best_match = None
                for match in matches:
                    parts = match.split(os.sep)
                    try:
                        epoch_val = int(parts[-2])
                        if epoch_val > highest_epoch:
                            highest_epoch = epoch_val
                            best_match = match
                    except ValueError:
                        continue
                
                if best_match:
                    dest_filename = f"{county}_simulation_results.png"
                    dest_path = os.path.join(dest_dir, dest_filename)
                    shutil.copy2(best_match, dest_path)
                    print(f"[{county}] Copied (fallback glob) from epoch {highest_epoch}")
                    copied_count += 1
                    found = True

            if not found:
                print(f"[{county}] WARNING: No simulation_results.png found under standard paths.")
                missing_count += 1

    print(f"\nDone! Successfully collected {copied_count} graphs.")
    if missing_count > 0:
        print(f"Notice: {missing_count} counties did not have completed simulation graphs.")

if __name__ == "__main__":
    main()
