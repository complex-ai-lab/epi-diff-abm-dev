import torch
import numpy as np
import re
import torch.nn.functional as F

from agent_torch.core.helpers import get_by_path
from agent_torch.core.substep import SubstepAction
# from agent_torch.core.llm.backend import LangchainLLM
from agent_torch.core.distributions import StraightThroughBernoulli


class MakeIsolationDecision(SubstepAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.mode = self.config["simulation_metadata"]["EXECUTION_MODE"]
        self.num_agents = self.config["simulation_metadata"]["num_agents"]

        self.st_bernoulli = StraightThroughBernoulli.apply

    def string_to_number(self, string):
        if string.lower() == "yes":
            return 1
        else:
            return 0

    def change_text(self, change_amount):
        change_amount = int(change_amount)
        if change_amount >= 1:
            return f"a {change_amount}% increase from last week"
        elif change_amount <= -1:
            return f"a {abs(change_amount)}% decrease from last week"
        else:
            return "the same as last week"

    def _generate_one_hot_tensor(self, timestep, num_timesteps):
        timestep_tensor = torch.tensor([timestep])
        one_hot_tensor = F.one_hot(timestep_tensor, num_classes=num_timesteps)

        return one_hot_tensor.to(self.device)

    def forward(self, state, observation):
        # if in debug mode, return random values for isolation
        md = self.config["simulation_metadata"]
        cf = str(md.get("GENERATING_COUNTERFACTUAL", False)).lower() in ("true", "1")
        if cf:
            # CRN: isolation draw keyed on (random_seed, _cf_iter, step) -- NOT
            # on COUNTERFACTUAL_TYPE -- so it is identical across CF policies.
            t = int(state["current_step"])
            base = int(md.get("random_seed", md.get("SEED", 42)))
            it = int(md.get("_cf_iter", 0))
            s = (((base & 0xFFFFF) * 1_000_003 + it) * 1_000_003 + t) * 131 + 23
            g = torch.Generator()
            g.manual_seed(int(s & 0x7FFF_FFFF_FFFF_FFFF))
            will_isolate = torch.rand(self.num_agents, 1, generator=g).to(self.device)
        else:
            will_isolate = torch.rand(self.num_agents, 1).to(self.device)

        return {self.output_variables[0]: will_isolate}
