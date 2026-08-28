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

# --- Resume support -----------------------------------------------------------
# The previous run was interrupted mid-way. This script now inspects the network
# output directory and skips states whose counties are ALL already fully
# generated, then resumes from the first state that is missing / incomplete.
#
# "To be safe, restart from the last (incomplete) state entirely": a state is
# only treated as done when every one of its counties is complete. The last
# state that has any output but is not fully complete is regenerated from
# scratch (generate_networks.sh always overwrites), together with every state
# after it.
#
# Environment overrides:
#   FORCE_ALL=1          -> ignore existing output, (re)run every state
#   START_STATE=IL       -> run this state and every later state, skip earlier
#   ONLY_STATES=IL,KS    -> run only these states (comma separated)
#   DRY_RUN=1            -> print the resume plan and exit without generating
# ---------------------------------------------------------------------------

# Files that must exist for a county's mobility networks to count as complete.
_REQUIRED_NETWORK_FILES = ("HOUSEHOLD_NETWORK", "SCHOOL_NETWORK")


def resolve_output_dir(project_root):
    """Mirror the "full" experiment output path from initialize_experiment.py."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(project_root, '.env'))
    except Exception:
        pass

    networks_dir = os.getenv("NETWORKS_DIR")
    if networks_dir:
        if not os.path.isabs(networks_dir):
            return os.path.join(project_root, networks_dir)
        return networks_dir
    return os.path.join(project_root, 'data', 'networks', 'covid_output_causal')


def _count_pkls(path):
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if f.endswith('.pkl') or f.endswith('.csv'))


def reference_counts(output_dir):
    """Largest occnets / randnets counts seen across all existing county dirs.

    Used as the expected per-county counts so completeness is self-calibrating
    and does not need to hardcode the number of timesteps.
    """
    max_occ = 0
    max_rand = 0
    if not os.path.isdir(output_dir):
        return max_occ, max_rand
    for name in os.listdir(output_dir):
        mob = os.path.join(output_dir, name, 'mobility_networks')
        max_occ = max(max_occ, _count_pkls(os.path.join(mob, 'occnets')))
        max_rand = max(max_rand, _count_pkls(os.path.join(mob, 'randnets')))
    return max_occ, max_rand


def county_is_complete(output_dir, county, ref_occ, ref_rand):
    mob = os.path.join(output_dir, county, 'mobility_networks')
    if not os.path.isdir(mob):
        return False
    for base in _REQUIRED_NETWORK_FILES:
        if not (os.path.exists(os.path.join(mob, base + '.pkl'))
                or os.path.exists(os.path.join(mob, base + '.csv'))):
            return False
    # occnets are generated first, randnets last; require both to match the
    # fullest county we have on disk.
    if ref_occ and _count_pkls(os.path.join(mob, 'occnets')) < ref_occ:
        return False
    if ref_rand and _count_pkls(os.path.join(mob, 'randnets')) < ref_rand:
        return False
    return ref_occ > 0 and ref_rand > 0


def state_status(output_dir, state_counties, ref_occ, ref_rand):
    done = [c for c in state_counties
            if county_is_complete(output_dir, c, ref_occ, ref_rand)]
    if len(done) == len(state_counties):
        return 'complete', done
    if done:
        return 'partial', done
    return 'missing', done


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = resolve_output_dir(project_root)

    force_all = os.getenv("FORCE_ALL") == "1"
    start_state = (os.getenv("START_STATE") or "").strip().upper() or None
    only_states = {s.strip().upper() for s in (os.getenv("ONLY_STATES") or "").split(",") if s.strip()}

    # Group counties by state
    state_to_counties = {}
    for county in COUNTIES:
        state_fips = county[:2]
        state_abbr = fips_to_state.get(state_fips)
        if not state_abbr:
            print(f"Warning: Unknown state FIPS prefix '{state_fips}' for county '{county}'. Skipping.")
            continue
        state_to_counties.setdefault(state_abbr, []).append(county)

    print(f"Loaded {len(COUNTIES)} unique counties across {len(state_to_counties)} states.")
    print(f"Network output directory: {output_dir}")

    ref_occ, ref_rand = reference_counts(output_dir)
    print(f"Reference per-county counts from existing output: occnets={ref_occ}, randnets={ref_rand}")

    ordered_states = [s for s, _ in sorted(state_to_counties.items())]

    # Decide which states to run and print a resume summary.
    to_run = []
    print("\n----------------------- Resume plan -----------------------")
    for state in ordered_states:
        state_counties = state_to_counties[state]
        status, done = state_status(output_dir, state_counties, ref_occ, ref_rand)

        if force_all:
            decision = "RUN (FORCE_ALL)"
            run_it = True
        elif only_states:
            run_it = state in only_states
            decision = "RUN (ONLY_STATES)" if run_it else "skip (not in ONLY_STATES)"
        elif start_state is not None:
            run_it = state >= start_state
            decision = "RUN (>= START_STATE)" if run_it else "skip (< START_STATE)"
        else:
            run_it = status != 'complete'
            decision = "RUN" if run_it else "skip (already complete)"

        if run_it:
            to_run.append(state)
        print(f"  {state}: {len(done)}/{len(state_counties)} counties done [{status}] -> {decision}")
    print("----------------------------------------------------------")

    if not to_run:
        print("\nNothing to do. All states already complete.")
        return

    print(f"\nStates to (re)generate: {', '.join(to_run)}\n")

    if os.getenv("DRY_RUN") == "1":
        print("DRY_RUN=1 set; exiting without generating networks.")
        return

    networks_dir = os.path.join(project_root, 'networks')
    script_path = os.path.join(networks_dir, 'generate_networks.sh')

    # Run generate_networks.sh for each state that needs it
    for state in to_run:
        state_counties = state_to_counties[state]
        counties_str = ','.join(state_counties)
        print(f"\n=======================================================")
        print(f"Generating networks for State: {state}")
        print(f"Counties ({len(state_counties)}): {counties_str}")
        print(f"=======================================================")

        env = os.environ.copy()
        env["TARGET_STATE"] = state
        env["TARGET_COUNTIES"] = counties_str

        # Run generate_networks.sh inside the networks directory
        result = subprocess.run(['bash', script_path], cwd=networks_dir, env=env)
        if result.returncode != 0:
            print(f"Error: generate_networks.sh failed for state {state} (exit code: {result.returncode})")
        else:
            print(f"Successfully completed network generation for state {state}")


if __name__ == '__main__':
    main()
