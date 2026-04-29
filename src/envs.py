import gymnasium as gym
from gymnasium.wrappers import TimeLimit

from configs import CaseConfig
from safety_wrappers import HolePenaltyWrapper


def make_env(
    case: CaseConfig,
    safe_reward: bool = False,
    hole_penalty: float = -1.0,
):
    """
    Crée un environnement FrozenLake pour un cas d'étude donné.

    Args:
        case: configuration du cas d'étude.
        safe_reward: si True, ajoute une pénalité lorsque l'agent tombe dans un trou.
        hole_penalty: pénalité appliquée aux trous pour DQN-Safe.

    Returns:
        Environnement Gymnasium.
    """

    env = gym.make(
        "FrozenLake-v1",
        desc=case.desc,
        is_slippery=case.is_slippery,
        success_rate=case.success_rate,
    )

    max_steps = 100 if len(case.desc) == 4 else 200
    env = TimeLimit(env, max_episode_steps=max_steps)

    if safe_reward:
        env = HolePenaltyWrapper(env, hole_penalty=hole_penalty)

    return env