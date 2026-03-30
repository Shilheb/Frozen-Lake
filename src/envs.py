import gymnasium as gym

from configs import CaseConfig


def _get_max_episode_steps(case_name: str) -> int:
    if "8x8" in case_name:
        return 200
    return 100


def make_env(case_cfg: CaseConfig):
    env = gym.make(
        "FrozenLake-v1",
        desc=case_cfg.desc,
        is_slippery=case_cfg.is_slippery,
        success_rate=case_cfg.success_rate,
    )

    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=_get_max_episode_steps(case_cfg.name),
    )
    return env