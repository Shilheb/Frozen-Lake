import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from configs import CaseConfig


def make_env(case: CaseConfig):
    env = gym.make(
        "FrozenLake-v1",
        desc=case.desc,
        is_slippery=case.is_slippery,
        success_rate=case.success_rate,
    )
    max_steps = 100 if len(case.desc) == 4 else 200
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env
