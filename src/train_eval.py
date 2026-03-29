from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from configs import ALGO_CONFIGS, CASE_CONFIGS, EVAL_EPISODES, EVAL_INTERVAL, SEEDS, TOTAL_TIMESTEPS
from envs import make_env
from metrics import evaluate_model

# Imports kept inside function to make module import cheaper when dependencies are missing.


def _build_model(algo_name: str, policy: str, env, seed: int, params: Dict):
    if algo_name == "DQN":
        from stable_baselines3 import DQN
        return DQN(policy, env, seed=seed, **params)
    if algo_name == "PPO":
        from stable_baselines3 import PPO
        return PPO(policy, env, seed=seed, **params)
    raise ValueError(f"Unsupported algorithm: {algo_name}")


def run_experiment(output_dir: str = "results") -> pd.DataFrame:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []

    for case in CASE_CONFIGS:
        for algo in ALGO_CONFIGS:
            total_timesteps = TOTAL_TIMESTEPS[case.name]

            for seed in SEEDS:
                env = make_env(case)
                model = _build_model(algo.name, algo.policy, env, seed, algo.params.copy())

                for current_step in range(EVAL_INTERVAL, total_timesteps + 1, EVAL_INTERVAL):
                    model.learn(total_timesteps=EVAL_INTERVAL, reset_num_timesteps=False, progress_bar=False)
                    eval_env = make_env(case)
                    metrics = evaluate_model(model, eval_env, case.desc, n_episodes=EVAL_EPISODES)
                    row = {
                        "case": case.name,
                        "algorithm": algo.name,
                        "seed": seed,
                        "timesteps": current_step,
                        **metrics,
                    }
                    all_rows.append(row)
                    eval_env.close()

                env.close()

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "metrics_long.csv", index=False)

    summary = (
        df.sort_values("timesteps")
        .groupby(["case", "algorithm", "seed"], as_index=False)
        .tail(1)
        .groupby(["case", "algorithm"], as_index=False)
        .agg(
            eval_return_mean=("eval_return_mean", "mean"),
            eval_success_rate=("eval_success_rate", "mean"),
            eval_hole_rate=("eval_hole_rate", "mean"),
            eval_episode_length_mean=("eval_episode_length_mean", "mean"),
        )
    )
    summary.to_csv(out / "summary.csv", index=False)
    return df
