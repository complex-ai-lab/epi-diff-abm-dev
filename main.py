#!/usr/bin/env python3
import abm_nets
from agent_torch.core.executor import Executor
from agent_torch.core.dataloader import LoadPopulation
import covid_abm
from populations import pop01045

def main():
    sim = Executor(covid_abm, pop_loader=LoadPopulation(pop01045))
    runner = sim._get_runner(sim.config)
    runner.init()
    
    abm_nets.eval_net(sim, runner)

if __name__ == '__main__':
    main()
