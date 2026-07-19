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
import glob

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
    metro_phase = meta.get("metro_calibration_phase", 0)
    use_7day_avg = meta.get("USE_7DAY_AVG", True)

    print("--- Simulation Parameters Detected ---")
    print(f"Date:                  {date}")
    print(f"Initial Infection Rate:{initial_rate}")
    print(f"Exposed to Infected:   {exposed_to_infected}")
    print(f"Infected to Recovered: {infected_to_recovered}")
    print(f"With K:                {with_k}")
    print(f"Metro Calibration Phase:{metro_phase}")
    print(f"Use 7-Day Average:     {use_7day_avg}")
    print(f"Destination Directory:  {dest_dir}")
    print("---------------------------------------\n")

    copied_count = 0
    missing_count = 0

    # Define the target folders and epochs we want to pull per county
    targets = [
        {"folder": "0.0005_3_5_True_False_metro_0", "epoch": 250},
        {"folder": "0.0005_3_5_True_True_metro_0", "epoch": 250},
        {"folder": "0.0005_3_5_True_True_metro_0", "epoch": 500},
        {"folder": "0.0005_3_5_True_True_False_metro_0", "epoch": 250},
        {"folder": "0.0005_3_5_True_True_False_metro_0", "epoch": 500},
    ]

    for county in COUNTIES:
        for target in targets:
            folder_name = target["folder"]
            epoch_val = target["epoch"]
            epochs_to_check = [epoch_val, epoch_val - 1]
            
            found_target = False
            
            # 1. Try exact folder name and epoch
            for ep in epochs_to_check:
                src_path = os.path.join(
                    "result_graphs",
                    county,
                    date,
                    folder_name,
                    str(ep),
                    "simulation_results.png"
                )
                if os.path.exists(src_path):
                    dest_filename = f"{county}_{folder_name}_epoch_{epoch_val}.png"
                    dest_path = os.path.join(dest_dir, dest_filename)
                    shutil.copy2(src_path, dest_path)
                    print(f"[{county}] Copied target '{folder_name}' epoch {epoch_val} (found at epoch {ep})")
                    copied_count += 1
                    found_target = True
                    break

            # 2. Fallback: try phase variations (e.g. if folder_name ends with _metro_0, check _metro_1, _metro_2)
            if not found_target and "_metro_" in folder_name:
                base_folder, curr_phase = folder_name.rsplit("_metro_", 1)
                phases_to_check = [curr_phase] + [str(p) for p in [0, 1, 2] if str(p) != curr_phase]
                for phase in phases_to_check:
                    alt_folder = f"{base_folder}_metro_{phase}"
                    for ep in epochs_to_check:
                        src_path = os.path.join(
                            "result_graphs",
                            county,
                            date,
                            alt_folder,
                            str(ep),
                            "simulation_results.png"
                        )
                        if os.path.exists(src_path):
                            dest_filename = f"{county}_{folder_name}_epoch_{epoch_val}.png"
                            dest_path = os.path.join(dest_dir, dest_filename)
                            shutil.copy2(src_path, dest_path)
                            print(f"[{county}] Copied target '{folder_name}' epoch {epoch_val} from fallback folder '{alt_folder}' epoch {ep}")
                            copied_count += 1
                            found_target = True
                            break
                    if found_target:
                        break

            # 3. Fallback glob search
            if not found_target:
                for ep in epochs_to_check:
                    glob_pattern = os.path.join("result_graphs", county, date, f"{folder_name}*", str(ep), "simulation_results.png")
                    matches = glob.glob(glob_pattern)
                    if matches:
                        src_path = matches[0]
                        dest_filename = f"{county}_{folder_name}_epoch_{epoch_val}.png"
                        dest_path = os.path.join(dest_dir, dest_filename)
                        shutil.copy2(src_path, dest_path)
                        parts = src_path.split(os.sep)
                        matched_folder = parts[-3]
                        print(f"[{county}] Copied (glob) target '{folder_name}' epoch {epoch_val} from folder '{matched_folder}' epoch {ep}")
                        copied_count += 1
                        found_target = True
                        break

            if not found_target:
                print(f"[{county}] WARNING: No simulation_results.png found for target '{folder_name}' epoch {epoch_val}")
                missing_count += 1

    print(f"\nDone! Successfully collected {copied_count} graphs.")
    if missing_count > 0:
        print(f"Notice: {missing_count} targets were missing.")

if __name__ == "__main__":
    main()
