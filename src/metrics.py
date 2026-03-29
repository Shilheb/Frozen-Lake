from typing import Dict, List
import numpy as np


def tile_type_from_state(desc: List[str], state: int) -> str:
    ncol = len(desc[0])
    row, col = divmod(state, ncol)
    return desc[row][col]


def evaluate_model(model, env, desc: List[str], n_episodes: int = 200) -> Dict[str, float]:
    rewards = []
    successes = []
    holes = []
    lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        fell_in_hole = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)

            # convertir l'action en int Python
            if isinstance(action, np.ndarray):
                action = int(action.item())
            else:
                action = int(action)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if terminated and reward == 0:
                if tile_type_from_state(desc, int(obs)) == "H":
                    fell_in_hole = 1

        rewards.append(total_reward)
        successes.append(1 if total_reward > 0 else 0)
        holes.append(fell_in_hole)
        lengths.append(steps)

    return {
        "eval_return_mean": float(np.mean(rewards)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_hole_rate": float(np.mean(holes)),
        "eval_episode_length_mean": float(np.mean(lengths)),
    }