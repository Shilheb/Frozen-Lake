from typing import Dict, List

import numpy as np
import pandas as pd


def _to_discrete_action(action) -> int:
    """
    Convertit l'action retournée par SB3 en entier Python compatible
    avec FrozenLake (espace d'actions discret).
    """
    if isinstance(action, np.ndarray):
        if action.size == 1:
            return int(action.item())
        raise ValueError(f"Unexpected action shape for discrete env: {action.shape}")
    return int(action)


def evaluate_model(model, env, n_episodes: int) -> Dict[str, float]:
    successes = []
    returns = []
    holes = []
    lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0.0
        episode_length = 0
        fell_in_hole = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action = _to_discrete_action(action)

            obs, reward, done, truncated, _ = env.step(action)

            episode_return += float(reward)
            episode_length += 1

            if done and reward == 0.0:
                fell_in_hole = 1

        successes.append(1 if episode_return > 0 else 0)
        returns.append(episode_return)
        holes.append(fell_in_hole)
        lengths.append(episode_length)

    return {
        "success_rate": float(np.mean(successes)),
        "avg_return": float(np.mean(returns)),
        "hole_rate": float(np.mean(holes)),
        "avg_episode_length": float(np.mean(lengths)),
    }


def aggregate_with_confidence_intervals(raw_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "success_rate",
        "avg_return",
        "hole_rate",
        "avg_episode_length",
    ]

    grouped = raw_df.groupby(["case", "algo", "timestep"], as_index=False)

    rows: List[Dict[str, float]] = []
    for (case_name, algo_name, timestep), group in grouped:
        row: Dict[str, float] = {
            "case": case_name,
            "algo": algo_name,
            "timestep": int(timestep),
            "n_seeds": int(len(group)),
        }

        for metric in metric_cols:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci95 = float(1.96 * std / np.sqrt(len(values))) if len(values) > 1 else 0.0

            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = ci95

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["case", "algo", "timestep"]).reset_index(drop=True)