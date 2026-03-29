# FrozenLake preliminary analysis

This repository contains the **preliminary analysis** for a reinforcement learning project on **FrozenLake-v1**.  
The goal is to compare two baseline reinforcement learning strategies, **DQN** and **PPO**, on multiple FrozenLake case studies with different difficulty levels and transition dynamics.

The project is organized to make the experiments easy to reproduce, evaluate, and analyze. It includes:
- centralized experiment configuration
- environment creation
- training and evaluation loops
- metric computation
- automatic figure generation
- CSV export of results

---

## Project objective

The purpose of this project is to perform an initial empirical study of reinforcement learning algorithms on FrozenLake.  
More specifically, the project compares **DQN** and **PPO** on three case studies:
- a deterministic 4x4 map
- a slippery 4x4 map
- a harder slippery 8x8 map

For each case, the code trains the selected algorithms, evaluates them periodically during training, saves the metrics, and generates plots for comparison.

---

## Repository structure

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

Description of each file
Root files
README.md

This file explains the project, the repository structure, how to install dependencies, how to run the code, what outputs are generated, and what each source file does.

requirements.txt

This file lists the Python dependencies required to run the project:

gymnasium
stable-baselines3
numpy
pandas
matplotlib
methodology_notes.md

This file contains methodology notes for the report, including:

libraries used
algorithm hyperparameters
experimental protocol
metrics
reproducibility notes
.gitignore

This file prevents unnecessary files from being tracked by Git, such as:

Python cache files
virtual environment folders
generated results
generated figures
Source files in src/
configs.py

This file centralizes the experimental configuration of the project.
It defines:

the case studies
the algorithm configurations
the list of seeds
the total training timesteps for each case
the evaluation interval
the number of evaluation episodes

It is the main configuration file of the project.

envs.py

This file creates the FrozenLake environments used in the experiments.
It:

builds the environment from the selected case
sets the map description
sets whether the environment is slippery
sets the success rate
wraps the environment with a TimeLimit

The maximum episode length is:

100 steps for 4x4 maps
200 steps for 8x8 maps
metrics.py

This file contains the evaluation utilities.
It evaluates a trained model over multiple episodes and computes:

average episodic return
success rate
hole rate
average episode length

It also identifies whether an episode ended in a hole.

plotting.py

This file generates figures from the experiment logs.
It:

aggregates metrics across seeds
plots the selected metric as a function of timesteps
saves each plot as a PNG image

The plots are saved in the figures/ directory.

train_eval.py

This file contains the core experimental pipeline.
It:

creates the environments
builds the models
trains the algorithms
evaluates them periodically
stores all intermediate evaluation results
saves detailed and summary CSV files

It is responsible for the full train/evaluate/save workflow.

run_all.py

This file is the main entry point of the project.
Running it launches the full experiment:

training
evaluation
metric logging
figure generation

It also prints confirmation messages at the end.

Experimental setup
1. Case studies

The project currently includes three FrozenLake case studies.

Case 1: case1_deterministic_4x4
4x4 map
deterministic environment
is_slippery = False
success rate = 1.0

Map:

SFFF
FHFH
FFFH
FFFG
Case 2: case2_slippery_4x4
4x4 map
slippery environment
is_slippery = True
success rate = 1/3

Map:

SFFF
FHFH
FFFH
HFFG
Case 3: case3_hard_slippery_8x8
8x8 map
harder slippery environment
is_slippery = True
success rate = 0.25

Map:

SFFFFFFF
FHFHFFFF
FFFHFFHF
FHFHFFFF
FFFHHFHF
FHFHFFFF
FFFHHFHF
FFFFHFFG
2. Algorithms

The current implementation compares two baseline algorithms from Stable-Baselines3.

DQN
policy: MlpPolicy

Hyperparameters:

learning_rate = 1e-3
buffer_size = 50000
learning_starts = 1000
batch_size = 64
tau = 1.0
gamma = 0.99
train_freq = 4
gradient_steps = 1
target_update_interval = 1000
exploration_fraction = 0.30
exploration_initial_eps = 1.0
exploration_final_eps = 0.05
verbose = 0
PPO
policy: MlpPolicy

Hyperparameters:

learning_rate = 3e-4
n_steps = 512
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
verbose = 0
3. Training budget

The number of training timesteps depends on the case study:

Case 1: 30000
Case 2: 60000
Case 3: 120000
4. Evaluation protocol

The current code uses the following protocol:

evaluation every 2000 timesteps
200 evaluation episodes at each checkpoint
current seed list: [0]
5. Episode duration

The environment is wrapped with a time limit:

100 steps maximum for 4x4 maps
200 steps maximum for 8x8 maps
6. Metrics

The project computes the following evaluation metrics:

Average episodic return
Success rate
Hole rate
Average episode length

These metrics are saved to CSV files and used for figure generation.

Installation
1. Clone the repository
git clone https://github.com/Shilheb/Frozen-Lake.git
cd Frozen-Lake
2. Install the dependencies
python -m pip install -r requirements.txt
Dependencies

The required packages are:

gymnasium>=1.1
stable-baselines3>=2.3
numpy>=2.0
pandas>=2.2
matplotlib>=3.10

They are already listed in requirements.txt.

How to run the project

The current project is designed to be run from the src/ folder.

Step-by-step execution
Step 1: go to the repository folder
cd Frozen-Lake
Step 2: install dependencies
python -m pip install -r requirements.txt
Step 3: go to the source folder
cd src
Step 4: run the main script
python run_all.py
What happens when the project runs

When run_all.py is executed, the following steps are performed:

the experimental pipeline is launched
each case study is created
each algorithm is trained on the selected case
evaluation is performed every 2000 timesteps
the metrics are collected and stored
CSV files are saved in the results/ directory
figures are generated and saved in the figures/ directory

At the end of execution, the script prints confirmation messages indicating that:

the experiments are finished
results were saved in ../results
figures were saved in ../figures
Output files

After execution, the project generates output folders at the repository root.

results/

This folder contains:

metrics_long.csv

Detailed evaluation logs for all checkpoints.

This file includes:

case name
algorithm name
seed
timesteps
average return
success rate
hole rate
average episode length
summary.csv

Final aggregated metrics for each case and algorithm.

figures/

This folder contains generated PNG figures for:

success rate
hole rate
average return

Each figure is generated separately for each case study.


    
