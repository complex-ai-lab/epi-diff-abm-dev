#!/usr/bin/env python3
import sys
import os
import importlib
import abm_nets
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
import covid_abm

def run_county_phase(fips_code, phase):
    try:
        # Dynamically import the populations.pop<FIPS> module
        pop_module = importlib.import_module(f"populations.pop{fips_code}")
    except ModuleNotFoundError:
        print(f"Error: Population module populations.pop{fips_code} not found.")
        sys.exit(1)
        
    print(f"\n=======================================================")
    print(f"Running FIPS {fips_code} | Metro Calibration Phase {phase}")
    print(f"=======================================================")
    
    sim = Executor(covid_abm, pop_loader=LoadPopulation(pop_module))
    
    # Configure simulation metadata in-memory for this FIPS and phase
    sim.config['simulation_metadata']['metro_calibration_phase'] = phase
    sim.config['simulation_metadata']['POPULATION'] = fips_code
    
    # Update the population directory to point to the current FIPS code folder
    original_pop_dir = sim.config['simulation_metadata']['population_dir']
    parent_dir = os.path.dirname(original_pop_dir)
    sim.config['simulation_metadata']['population_dir'] = os.path.join(parent_dir, f"pop{fips_code}")
    
    runner = sim._get_runner(sim.config)
    runner.init()
    
    abm_nets.eval_net(sim, runner)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <fips_code_1,fips_code_2,...>")
        sys.exit(1)
        
    fips_input = sys.argv[1]
    fips_list = [f.strip() for f in fips_input.split(',')]
    fips_list = [f for f in fips_list if f]  # Remove empty entries
    
    if len(fips_list) == 1:
        run_county_phase(fips_list[0], phase=0)
    elif len(fips_list) > 1:
        # Phase 1: Factoring/Calibration Phase
        for fips in fips_list:
            run_county_phase(fips, phase=1)
            
        # Phase 2: Calibration with Commuter Interaction
        for fips in fips_list:
            run_county_phase(fips, phase=2)
    else:
        print("Error: No valid FIPS codes provided.")
        sys.exit(1)

if __name__ == '__main__':
    main()
