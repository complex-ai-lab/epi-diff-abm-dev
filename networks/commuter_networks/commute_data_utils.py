import os
import pandas as pd
import glob
import re
from tqdm import tqdm
import shutil
import numpy as np

def track_commute_data(individuals_df, state_abbrev, county, output_dir):
    # Find commute data relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    commute_data_path = os.path.join(script_dir, "commute_data", "formatted_residence_commute_data.csv")
    commute_df = pd.read_csv(commute_data_path, dtype={'residence_county': str, 'work_county': str})

    # Create dictionary for this county where key is workplace county and value is number of commuters
    this_county_commute_dict = {}
    for _, row in commute_df.iterrows():
        if row['residence_county'] == county:
            work_county = row['work_county']
            if work_county not in this_county_commute_dict:
                this_county_commute_dict[work_county] = 0
            this_county_commute_dict[work_county] += row['num_commuters']

    # Filter individuals with non-empty Occupations to select as commuters
    available_individuals = individuals_df[individuals_df['Occupations'].notna() & 
                                             (individuals_df['Occupations'] != '')].copy()

    # The occupation networks directory path as defined in gen_mob_nw.py
    output_occ_dir = os.path.join(output_dir, "mobility_networks", "occnets")
    os.makedirs(output_occ_dir, exist_ok=True)

    # Create output file for this county (residence perspective) in the occupation networks directory
    residence_output_path = os.path.join(output_occ_dir, f"{county}_residence_commute_data.txt")
    
    with open(residence_output_path, 'w') as f:
        for work_county, num_commuters in this_county_commute_dict.items():
            # Uniformly randomly select num_commuters agents from available individuals with occupations
            if len(available_individuals) >= num_commuters:
                selected_agents = np.random.choice(available_individuals['ID'].values, 
                                                   size=num_commuters, 
                                                   replace=False)
            else:
                # If we don't have enough agents, take all available agents
                selected_agents = available_individuals['ID'].values
            
            # Write to residence file (agent_id,work_county)
            for agent_id in selected_agents:
                f.write(f"{agent_id},{work_county}\n")
                
            # Remove selected agents from the pool of available individuals for this county
            available_individuals = available_individuals[~available_individuals['ID'].isin(selected_agents)]
