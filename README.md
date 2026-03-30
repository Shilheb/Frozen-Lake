# FrozenLake Preliminary Analysis

This repository contains the **preliminary analysis** for a reinforcement learning project on **FrozenLake-v1**.
The goal is to compare two baseline reinforcement learning strategies, **DQN** and **PPO**, on multiple FrozenLake case studies with different difficulty levels and transition dynamics.

The project is organized to make the experiments easy to reproduce, evaluate, and analyze.

It includes:
- Centralized experiment configuration
- Environment creation
- Training and evaluation loops
- Metric computation
- Automatic figure generation
- CSV export of results

---

## Project Objective

The purpose of this project is to perform an initial empirical study of reinforcement learning algorithms on FrozenLake.

More specifically, the project compares **DQN** and **PPO** on three case studies:
- A deterministic 4×4 map
- A slippery 4×4 map
- A harder slippery 8×8 map

For each case, the code trains the selected algorithms, evaluates them periodically during training, saves the metrics, and generates plots for comparison.

---

## Repository Structure

```text
Frozen-Lake/
├── .gitignore
├── README.md
├── methodology_notes.md
├── requirements.txt
└── src/
    ├── configs.py
    ├── envs.py
    ├── metrics.py
    ├── plotting.py
    ├── run_all.py
    └── train_eval.py
```

---

## Description of Each File

### Root Files

#### `README.md`
This file explains the project, the repository structure, how to install dependencies, how to run the code, what outputs are generated, and what each source file does.

#### `requirements.txt`
Lists the Python dependencies required to run the project:
- gymnasium
- stable-baselines3
- numpy
- pandas
- matplotlib

#### `.gitignore`
Prevents unnecessary files from being tracked by Git, such as:
- Python cache files
- Virtual environment folders
- Generated outputs

---

### Source Files in `src/`

#### `configs.py`
Centralizes the experimental configuration of the project. It defines:
- The case studies
- The algorithm configurations
- The list of seeds
- The total training timesteps for each case
- The evaluation interval
- The number of evaluation episodes

#### `envs.py`
Creates the FrozenLake environments used in the experiments. It:
- Builds the environment from the selected case
- Sets the map description
- Sets whether the environment is slippery
- Sets the success rate
- Wraps the environment with a `TimeLimit`

Maximum episode length:
- **100 steps** for 4×4 maps
- **200 steps** for 8×8 maps

#### `metrics.py`
Contains the evaluation utilities. It evaluates a trained model over multiple episodes and computes:
- Average episodic return
- Success rate
- Hole rate
- Average episode length

#### `plotting.py`
Generates figures from the experiment logs. It:
- Aggregates metrics across seeds
- Plots the selected metric as a function of timesteps
- Saves each plot as a PNG image

#### `train_eval.py`
Contains the core experimental pipeline. It:
- Creates the environments
- Builds the models
- Trains the algorithms
- Evaluates them periodically
- Stores all intermediate evaluation results
- Saves detailed and aggregated CSV files

#### `run_all.py`
The main entry point of the project. Running it launches the full experiment:
- Training
- Evaluation
- Metric logging
- Figure generation
- Policy visualization

---

## Experimental Setup

### 1. Case Studies

#### Case 1 — `case1_deterministic_4x4`
- 4×4 map
- Deterministic environment (`is_slippery = False`)
- Success rate = `1.0`

```
SFFF
FHFH
FFFH
FFFG
```

#### Case 2 — `case2_slippery_4x4`
- 4×4 map
- Slippery environment (`is_slippery = True`)
- Success rate = `1/3`

```
SFFF
FHFH
FFFH
HFFG
```

#### Case 3 — `case3_hard_slippery_8x8`
- 8×8 map
- Harder slippery environment (`is_slippery = True`)
- Success rate = `0.25`

```
SFFFFFFF
FHFHFFFF
FFFHFFHF
FHFHFFFF
FFFHHFHF
FHFHFFFF
FFFHHFHF
FFFFHFFG
```

---

### 2. Algorithms

The project compares two baseline algorithms from **Stable-Baselines3**.

#### DQN
- Policy: `MlpPolicy`

| Hyperparameter | Value |
|---|---|
| learning_rate | `1e-3` |
| buffer_size | `50000` |
| learning_starts | `1000` |
| batch_size | `64` |
| tau | `1.0` |
| gamma | `0.99` |
| train_freq | `4` |
| gradient_steps | `1` |
| target_update_interval | `1000` |
| exploration_fraction | `0.30` |
| exploration_initial_eps | `1.0` |
| exploration_final_eps | `0.05` |
| verbose | `0` |

#### PPO
- Policy: `MlpPolicy`

| Hyperparameter | Value |
|---|---|
| learning_rate | `3e-4` |
| n_steps | `512` |
| batch_size | `64` |
| n_epochs | `10` |
| gamma | `0.99` |
| gae_lambda | `0.95` |
| clip_range | `0.2` |
| ent_coef | `0.01` |
| vf_coef | `0.5` |
| max_grad_norm | `0.5` |
| verbose | `0` |

---

### 3. Training Budget

| Case | Timesteps |
|---|---|
| Case 1 — Deterministic 4×4 | `30000` |
| Case 2 — Slippery 4×4 | `60000` |
| Case 3 — Hard Slippery 8×8 | `120000` |

---

### 4. Evaluation Protocol

- Evaluation every **2000 timesteps**
- **200 evaluation episodes** at each checkpoint
- Seed list: `[0, 1, 2, 3, 4]`
- Aggregation across seeds with **95% confidence intervals**

---

### 5. Episode Duration

| Map size | Max steps |
|---|---|
| 4×4 | 100 |
| 8×8 | 200 |

---

### 6. Metrics

The project computes the following evaluation metrics:
- Average episodic return
- Success rate
- Hole rate
- Average episode length

---

## Installation

```bash
git clone https://github.com/Shilheb/Frozen-Lake.git
cd Frozen-Lake
python -m pip install -r requirements.txt
```

---

## How to Run the Project

```bash
cd Frozen-Lake/src
python run_all.py
```

---

## Generated Outputs

Running `run_all.py` creates the following outputs under `../outputs/`:

- `raw_metrics.csv`: detailed evaluation logs for all seeds and checkpoints
- `aggregated_metrics.csv`: aggregated metrics with confidence intervals
- `figures/`: metric plots
- `policies/`: policy comparison visualizations

---

## Reproducibility Notes

The project is structured to make reproduction easier:
- Experimental settings are centralized in `src/configs.py`
- Environment definitions are explicitly listed in the configuration
- Algorithms and hyperparameters are centralized
- The evaluation protocol is fixed in the code
- Metrics are exported to CSV files
- Plots and policy visualizations are generated automatically

The current repository state is consistent with the code in `src/configs.py` and `src/run_all.py`.

---

## Notes for the Professor

This repository contains the preliminary analysis code for the FrozenLake reinforcement learning project.

It includes:
- The experimental setup
- The baseline methods
- The execution steps
- The generated outputs
- The role of each file in the repository

The project is organized so that the code, outputs, and methodology are easy to understand and reproduce.
