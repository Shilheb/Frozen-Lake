import random
from typing import Dict, List, Type

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN, PPO

from configs import EVAL_EPISODES, EVAL_INTERVAL, TOTAL_TIMESTEPS, AlgoConfig, CaseConfig
from envs import make_env
from metrics import evaluate_model


ALGO_CLASS = {
    "DQN": DQN,
    "PPO": PPO,
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_and_evaluate_case(case_cfg: CaseConfig, algo_cfg: AlgoConfig, seed: int):
    set_global_seed(seed)

    train_env = make_env(case_cfg)
    eval_env = make_env(case_cfg)

    algo_class: Type = ALGO_CLASS[algo_cfg.name]
    model = algo_class(
        policy=algo_cfg.policy,
        env=train_env,
        seed=seed,
        **algo_cfg.params,
    )

    total_timesteps = TOTAL_TIMESTEPS[case_cfg.name]
    evaluation_steps: List[int] = list(range(EVAL_INTERVAL, total_timesteps + 1, EVAL_INTERVAL))

    rows: List[Dict] = []

    learned_so_far = 0
    for step_target in evaluation_steps:
        additional_steps = step_target - learned_so_far
        model.learn(total_timesteps=additional_steps, reset_num_timesteps=False, progress_bar=False)
        learned_so_far = step_target

        metrics = evaluate_model(model, eval_env, n_episodes=EVAL_EPISODES)

        row = {
            "case": case_cfg.name,
            "algo": algo_cfg.name,
            "seed": seed,
            "timestep": step_target,
        }
        row.update(metrics)
        rows.append(row)

    train_env.close()
    eval_env.close()

    return pd.DataFrame(rows), model