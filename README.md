# Safe Reinforcement Learning in FrozenLake

This repository contains the code for a reinforcement learning project on **safe learning in non-deterministic environments**, using `FrozenLake-v1` from **Gymnasium**.

The objective is to study how different reinforcement learning strategies behave when the agent must not only reach a goal, but also avoid dangerous states during learning. In FrozenLake, the dangerous states are the holes (`H`), which represent costly failures.

The project compares three strategies:

- **DQN**: baseline value-based deep RL method
- **PPO**: baseline policy-gradient / actor-critic method
- **DQN-Safe**: a safety-oriented variant of DQN using a modified reward that penalizes transitions into holes

The experiments are conducted on three FrozenLake case studies with increasing difficulty, from a deterministic 4×4 map to a harder stochastic 8×8 map.

---

## Project Objective

The goal of this project is to evaluate the impact of non-deterministic transition dynamics on the ability of reinforcement learning agents to avoid dangerous states during training.

More specifically, the study aims to answer the following questions:

1. How do standard RL baselines such as DQN and PPO behave when the environment becomes more stochastic?
2. Are success rate and return sufficient to evaluate performance in a safety-sensitive setting?
3. Can an explicit safety penalty reduce the number of dangerous failures during learning?
4. What trade-off appears between reaching the goal efficiently and avoiding holes?

To answer these questions, the project evaluates each strategy using multiple metrics:

- Average episodic return
- Success rate
- Hole rate
- Average episode length

The **hole rate** is the main safety indicator, since falling into a hole corresponds to entering a dangerous state.

---

## Repository Structure

```text
Frozen-Lake/
├── .gitignore
├── README.md
├── requirements.txt
├── outputs/
│   ├── raw_metrics.csv
│   ├── aggregated_metrics.csv
│   ├── figures/
│   └── policies/
└── src/
    ├── __init__.py
    ├── configs.py
    ├── envs.py
    ├── metrics.py
    ├── plotting.py
    ├── train_eval.py
    └── run_all.py
```

---

## Description of the Source Files

### `src/configs.py`

Centralizes the experimental configuration:

- Case-study definitions
- Map layouts
- Slippery / deterministic transition settings
- Success rates
- Training budgets
- Evaluation frequency
- Number of evaluation episodes
- Random seeds
- Algorithm hyperparameters

### `src/envs.py`

Creates the FrozenLake environments used in the experiments.

It is responsible for:

- Building the selected map
- Setting `is_slippery`
- Configuring the transition success rate
- Applying a maximum episode length through `TimeLimit`
- Creating the DQN-Safe reward wrapper when needed

Maximum episode duration:

| Map size | Maximum steps |
|---|---:|
| 4×4 | 100 |
| 8×8 | 200 |

### `src/metrics.py`

Contains the evaluation utilities.

For each trained model, the code evaluates the policy over several episodes and computes:

- Mean return
- Success rate
- Hole rate
- Mean episode length

### `src/plotting.py`

Generates figures from the saved experiment logs.

It supports:

- Aggregation across seeds
- Learning-curve visualization
- Confidence interval plotting
- Policy visualization

### `src/train_eval.py`

Contains the main training and evaluation pipeline.

It performs:

- Environment creation
- Model initialization
- Training
- Periodic evaluation
- Metric logging
- CSV export
- Policy extraction

### `src/run_all.py`

Main entry point of the project.

Running this file launches the complete experimental pipeline:

- Training all strategies
- Evaluating all checkpoints
- Saving raw and aggregated metrics
- Generating plots
- Generating policy visualizations

---

## Case Studies

The experiments use three FrozenLake configurations with increasing difficulty.

### Case 1 — Deterministic 4×4

This case is used as a controlled reference environment.

- Map size: 4×4
- `is_slippery = False`
- Success rate: `1.0`
- Maximum episode length: 100 steps

```text
SFFF
FHFH
FFFH
FFFG
```

### Case 2 — Slippery 4×4

This case introduces stochastic transitions. The agent may fail to execute the intended action, which makes paths near holes more dangerous.

- Map size: 4×4
- `is_slippery = True`
- Success rate: `1/3`
- Maximum episode length: 100 steps

```text
SFFF
FHFH
FFFH
HFFG
```

### Case 3 — Hard Slippery 8×8

This case combines a larger map, a longer horizon, more holes, and stronger transition uncertainty.

- Map size: 8×8
- `is_slippery = True`
- Success rate: `0.25`
- Maximum episode length: 200 steps

```text
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

## Compared Strategies

### DQN

DQN is a value-based off-policy deep reinforcement learning method.

It learns an approximation of the action-value function \(Q(s,a)\) using a neural network and updates it using a Bellman target. In this project, DQN serves as a baseline because it does not include an explicit mechanism for avoiding dangerous states.

DQN uses epsilon-greedy exploration. This allows the agent to explore the environment, but it can also lead to unsafe actions during learning.

### PPO

PPO is an on-policy actor-critic method.

It directly optimizes a stochastic policy while limiting overly large policy updates through a clipped objective. PPO is used as a second baseline because it represents a different family of RL algorithms from DQN.

Since PPO is on-policy, it relies heavily on recently collected trajectories. In sparse-reward and safety-sensitive environments, this can make learning difficult when many trajectories end in holes.

### DQN-Safe

DQN-Safe is a safety-oriented variant of DQN.

It keeps the same learning algorithm and hyperparameters as DQN, but modifies the training reward to penalize transitions into holes:

```text
+1  if the agent reaches the goal
-C  if the agent falls into a hole
 0  otherwise
```

In the experiments, the penalty is fixed to:

```text
C = 1
```

The goal of DQN-Safe is to reduce the number of dangerous failures during learning. However, this penalty can also make the learned policy more conservative, especially in difficult stochastic environments.

During evaluation, all strategies are evaluated using the same standard FrozenLake reward to ensure a fair comparison.

---

## Hyperparameters

### DQN and DQN-Safe

| Hyperparameter | Value |
|---|---:|
| Policy | `MlpPolicy` |
| Learning rate | `1e-3` |
| Buffer size | `50000` |
| Learning starts | `1000` |
| Batch size | `64` |
| Gamma | `0.99` |
| Train frequency | `4` |
| Gradient steps | `1` |
| Target update interval | `1000` |
| Exploration fraction | `0.30` |
| Initial epsilon | `1.0` |
| Final epsilon | `0.05` |
| Safety penalty `C` | `1` for DQN-Safe only |

### PPO

| Hyperparameter | Value |
|---|---:|
| Policy | `MlpPolicy` |
| Learning rate | `3e-4` |
| `n_steps` | `512` |
| Batch size | `64` |
| `n_epochs` | `10` |
| Gamma | `0.99` |
| GAE lambda | `0.95` |
| Clip range | `0.2` |
| Entropy coefficient | `0.01` |
| Value-function coefficient | `0.5` |
| Max gradient norm | `0.5` |

---

## Training Budget

| Case | Training timesteps |
|---|---:|
| Case 1 — Deterministic 4×4 | `30000` |
| Case 2 — Slippery 4×4 | `60000` |
| Case 3 — Hard Slippery 8×8 | `120000` |

---

## Evaluation Protocol

Each strategy is evaluated periodically during training.

| Parameter | Value |
|---|---:|
| Evaluation frequency | Every `2000` timesteps |
| Evaluation episodes | `200` episodes |
| Random seeds | `0, 1, 2, 3, 4` |
| Aggregation | Mean across seeds |
| Uncertainty | 95% confidence interval |

The confidence interval is computed across the five independent random seeds.

---

## Metrics

The project reports four metrics.

### Average episodic return

Measures the average cumulative reward obtained during evaluation episodes.

### Success rate

Measures the proportion of evaluation episodes in which the agent reaches the goal `G`.

### Hole rate

Measures the proportion of evaluation episodes in which the agent falls into a hole `H`.

This is the main safety metric.

### Average episode length

Measures how long episodes last on average.

This metric helps distinguish between:

- Fast successful policies
- Fast failures
- Conservative policies that survive but do not reach the goal efficiently

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Shilheb/Frozen-Lake.git
cd Frozen-Lake
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## How to Run the Experiments

From the repository root, run:

```bash
python src/run_all.py
```

Alternatively, if you are already inside `src/`, run:

```bash
python run_all.py
```

The script will train and evaluate all configured strategies on all case studies.

---

## Generated Outputs

After running the experiments, the following outputs are generated under the `outputs/` directory.

```text
outputs/
├── raw_metrics.csv
├── aggregated_metrics.csv
├── figures/
└── policies/
```

### `raw_metrics.csv`

Contains detailed evaluation results for each:

- Strategy
- Case study
- Seed
- Evaluation checkpoint

### `aggregated_metrics.csv`

Contains the metrics aggregated across seeds, including confidence intervals.

### `outputs/figures/`

Contains the learning curves for the evaluation metrics.

Examples:

- Success rate over training
- Hole rate over training
- Average return over training
- Average episode length over training

### `outputs/policies/`

Contains visualizations of the final learned policies.

The arrows represent the greedy action selected by the trained agent in each non-terminal state. Holes and goal states are not actionable.

---

## Reproducing the Report Results

To reproduce the results used in the report:

1. Install the dependencies.
2. Run the complete experiment pipeline:

```bash
python src/run_all.py
```

3. Check the generated CSV files in:

```text
outputs/
```

4. Use the figures in:

```text
outputs/figures/
outputs/policies/
```

The report figures are generated from the saved metrics and policy visualizations.

---

## Reproducibility Notes

The repository is designed to make the experiments reproducible:

- All case studies are defined explicitly in `src/configs.py`
- Random seeds are fixed
- Training budgets are fixed
- Evaluation frequency is fixed
- Evaluation episodes are fixed
- Hyperparameters are centralized
- Metrics are exported to CSV
- Figures are generated automatically

Because reinforcement learning training is stochastic, small numerical differences may occur depending on the machine, Python version, and installed package versions.

---

## Dependencies

The main dependencies are:

- `gymnasium`
- `stable-baselines3`
- `numpy`
- `pandas`
- `matplotlib`

Install them with:

```bash
python -m pip install -r requirements.txt
```

---

## Project Context

This project was developed for a reinforcement learning course at Université Laval.

The study focuses on the limitations of standard RL methods in safety-sensitive, non-deterministic environments and evaluates whether a simple safety penalty can reduce dangerous failures during learning.

---

## Authors

- Mohamed Gharbi  
  Université Laval  
  Maîtrise en informatique — Intelligence artificielle

- Syphax Ait Allak  
  Université Laval  
  Baccalauréat en informatique

---

## Repository

GitHub repository:

```text
https://github.com/Shilheb/Frozen-Lake
```
