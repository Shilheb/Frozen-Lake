from typing import Dict, List

import numpy as np
import pandas as pd


def tile_type_from_state(desc: List[str], state: int) -> str:
    """Retourne le type de case: S, F, H ou G."""
    n_cols = len(desc[0])
    row, col = divmod(int(state), n_cols)
    return desc[row][col]


def evaluate_model(
    model,
    env,
    desc: List[str],
    n_episodes: int = 200,
    deterministic: bool = True,
) -> Dict[str, float]:
    """
    Évalue une politique sur plusieurs épisodes.

    L'évaluation utilise l'environnement standard, pas la récompense pénalisée
    de DQN-Safe, afin de comparer équitablement les algorithmes.
    """

    rewards = []
    successes = []
    holes = []
    lengths = []

    for episode_idx in range(n_episodes):
        obs, _ = env.reset(seed=episode_idx)

        terminated = False
        truncated = False
        total_reward = 0.0
        steps = 0
        fell_in_hole = 0
        reached_goal = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action.item()) if isinstance(action, np.ndarray) else int(action)

            obs, reward, terminated, truncated, _ = env.step(action)

            total_reward += float(reward)
            steps += 1

            if terminated:
                tile = tile_type_from_state(desc, int(obs))
                if tile == "H":
                    fell_in_hole = 1
                elif tile == "G":
                    reached_goal = 1

        rewards.append(total_reward)
        successes.append(reached_goal)
        holes.append(fell_in_hole)
        lengths.append(steps)

    return {
        "eval_return_mean": float(np.mean(rewards)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_hole_rate": float(np.mean(holes)),
        "eval_episode_length_mean": float(np.mean(lengths)),
    }


def aggregate_with_confidence_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrège les résultats sur les graines et calcule un intervalle de confiance à 95%.

    Colonnes produites pour plotting.py:
        avg_return_mean / avg_return_ci95
        success_rate_mean / success_rate_ci95
        hole_rate_mean / hole_rate_ci95
        avg_episode_length_mean / avg_episode_length_ci95
    """

    group_cols = ["case", "algo", "timestep"]

    metric_map = {
        "eval_return_mean": "avg_return",
        "eval_success_rate": "success_rate",
        "eval_hole_rate": "hole_rate",
        "eval_episode_length_mean": "avg_episode_length",
    }

    rows = []

    for keys, group in df.groupby(group_cols):
        case, algo, timestep = keys
        n = group["seed"].nunique()

        row = {
            "case": case,
            "algo": algo,
            "timestep": timestep,
            "n_seeds": n,
        }

        for raw_metric, clean_name in metric_map.items():
            mean = group[raw_metric].mean()
            std = group[raw_metric].std(ddof=1)

            if n > 1 and not np.isnan(std):
                ci95 = 1.96 * std / np.sqrt(n)
            else:
                ci95 = 0.0

            row[f"{clean_name}_mean"] = float(mean)
            row[f"{clean_name}_std"] = float(std) if not np.isnan(std) else 0.0
            row[f"{clean_name}_ci95"] = float(ci95)

        rows.append(row)

    return pd.DataFrame(rows)
