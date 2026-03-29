# FrozenLake Preliminary Analysis

A small Python project for running **FrozenLake** reinforcement-learning experiments end to end: environment setup, training, periodic evaluation, metric export, and plot generation.

## What this project does

This repository compares RL agents on several predefined **FrozenLake-v1** case studies. The codebase is organized to support a simple experimental workflow:

1. define experiment cases and hyperparameters,
2. train agents,
3. evaluate them at regular intervals,
4. save metrics to CSV,
5. generate plots for analysis/reporting.

### Main outputs

Running the full pipeline produces:

- **tabular results** (CSV metrics over time and summary results),
- **plots** of key evaluation metrics such as success rate, hole rate, and return.

> **Assumption:** output folders are expected to be created under the project root as `results/` and `figures/`. If your local code currently writes elsewhere, update `src/run_all.py` or follow the printed paths at runtime.

---

## Project structure

```text
Frozen-Lake/
├── .gitignore
├── README.md
├── methodology_notes.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── configs.py
│   ├── envs.py
│   ├── metrics.py
│   ├── plotting.py
│   ├── train_eval.py
│   └── run_all.py