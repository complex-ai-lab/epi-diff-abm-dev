#!/usr/bin/env python3
import os
import subprocess

fips_to_state = {
    '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE',
    '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN',
    '19': 'IA', '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA',
    '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV',
    '33': 'NH', '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH',
    '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC', '46': 'SD', '47': 'TN',
    '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI',
    '56': 'WY'
}

# The same counties list as in run_all_prep.py
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

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Group counties by state
    state_to_counties = {}
    for county in COUNTIES:
        state_fips = county[:2]
        state_abbr = fips_to_state.get(state_fips)
        if not state_abbr:
            print(f"Warning: Unknown state FIPS prefix '{state_fips}' for county '{county}'. Skipping.")
            continue
        if state_abbr not in state_to_counties:
            state_to_counties[state_abbr] = []
        state_to_counties[state_abbr].append(county)
        
    print(f"Loaded {len(COUNTIES)} unique counties across {len(state_to_counties)} states.")
    
    # Run generate_networks.sh for each state
    for state, state_counties in sorted(state_to_counties.items()):
        counties_str = ','.join(state_counties)
        print(f"\n=======================================================")
        print(f"Generating networks for State: {state}")
        print(f"Counties ({len(state_counties)}): {counties_str}")
        print(f"=======================================================")
        
        env = os.environ.copy()
        env["TARGET_STATE"] = state
        env["TARGET_COUNTIES"] = counties_str
        
        networks_dir = os.path.join(project_root, 'networks')
        script_path = os.path.join(networks_dir, 'generate_networks.sh')
        
        # Run generate_networks.sh inside the networks directory
        result = subprocess.run(['bash', script_path], cwd=networks_dir, env=env)
        if result.returncode != 0:
            print(f"Error: generate_networks.sh failed for state {state} (exit code: {result.returncode})")
        else:
            print(f"Successfully completed network generation for state {state}")

if __name__ == '__main__':
    main()
