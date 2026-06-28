#!/usr/bin/env python3
import sys
import importlib
import abm_nets
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
import covid_abm

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 main.py <fips_code>")
        sys.exit(1)
        
    fips_code = sys.argv[1]
    
    try:
        # Dynamically import the populations.pop<FIPS> module
        pop_module = importlib.import_module(f"populations.pop{fips_code}")
    except ModuleNotFoundError:
        print(f"Error: Population module populations.pop{fips_code} not found.")
        sys.exit(1)
            
    sim = Executor(covid_abm, pop_loader=LoadPopulation(pop_module))
    runner = sim._get_runner(sim.config)
    runner.init()
    
    abm_nets.eval_net(sim, runner)

if __name__ == '__main__':
    main()
