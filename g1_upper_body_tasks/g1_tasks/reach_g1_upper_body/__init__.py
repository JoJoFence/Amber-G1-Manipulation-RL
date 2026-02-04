"""G1 Upper Body Reach Task"""

import gymnasium as gym

from . import agents
from .reach_env_cfg import G1UpperBodyReachEnvCfg

gym.register(
    id="Isaac-Reach-G1-Upper-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": G1UpperBodyReachEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}:G1UpperReachPPORunnerCfg",
    },
)
