from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class CaseConfig:
    name: str
    desc: List[str]
    is_slippery: bool
    success_rate: float


@dataclass
class AlgoConfig:
    name: str
    policy: str
    params: Dict[str, Any]


CASE_CONFIGS = [
    CaseConfig(
        name="case1_deterministic_4x4",
        desc=[
            "SFFF",
            "FHFH",
            "FFFH",
            "FFFG",
        ],
        is_slippery=False,
        success_rate=1.0,
    ),
    CaseConfig(
        name="case2_slippery_4x4",
        desc=[
            "SFFF",
            "FHFH",
            "FFFH",
            "HFFG",
        ],
        is_slippery=True,
        success_rate=1.0 / 3.0,
    ),
    CaseConfig(
        name="case3_hard_slippery_8x8",
        desc=[
            "SFFFFFFF",
            "FHFHFFFF",
            "FFFHFFHF",
            "FHFHFFFF",
            "FFFHHFHF",
            "FHFHFFFF",
            "FFFHHFHF",
            "FFFFHFFG",
        ],
        is_slippery=True,
        success_rate=0.25,
    ),
]

ALGO_CONFIGS = [
    AlgoConfig(
        name="DQN",
        policy="MlpPolicy",
        params=dict(
            learning_rate=1e-3,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            exploration_fraction=0.30,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            verbose=0,
        ),
    ),
    AlgoConfig(
        name="PPO",
        policy="MlpPolicy",
        params=dict(
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0,
        ),
    ),
]

SEEDS = [0, 1, 2, 3, 4]

TOTAL_TIMESTEPS = {
    "case1_deterministic_4x4": 30000,
    "case2_slippery_4x4": 60000,
    "case3_hard_slippery_8x8": 120000,
}

EVAL_INTERVAL = 2000
EVAL_EPISODES = 200

OUTPUT_DIR = "../outputs"
RAW_METRICS_FILENAME = "raw_metrics.csv"
AGG_METRICS_FILENAME = "aggregated_metrics.csv"