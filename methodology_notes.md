# Methodology notes for the report

## Libraries
- Gymnasium for FrozenLake-v1
- Stable-Baselines3 for DQN and PPO
- NumPy / pandas for data handling
- Matplotlib for plotting

## Hyperparameters
### DQN
- policy = MlpPolicy
- learning_rate = 1e-3
- buffer_size = 50000
- learning_starts = 1000
- batch_size = 64
- gamma = 0.99
- train_freq = 4
- gradient_steps = 1
- target_update_interval = 1000
- exploration_fraction = 0.30
- exploration_initial_eps = 1.0
- exploration_final_eps = 0.05

### PPO
- policy = MlpPolicy
- learning_rate = 3e-4
- n_steps = 512
- batch_size = 64
- n_epochs = 10
- gamma = 0.99
- gae_lambda = 0.95
- clip_range = 0.2
- ent_coef = 0.01
- vf_coef = 0.5
- max_grad_norm = 0.5

## Experimental protocol
- 5 random seeds
- Evaluation every 2,000 timesteps
- 200 evaluation episodes per checkpoint
- TimeLimit wrapper:
  - 100 steps for 4x4 maps
  - 200 steps for 8x8 maps
- Training budgets:
  - Case 1: 30,000 timesteps
  - Case 2: 60,000 timesteps
  - Case 3: 120,000 timesteps

## Metrics
- Average episodic return
- Success rate
- Hole rate
- Average episode length

## Reproducibility
- Fixed seeds
- All case definitions centralized in src/configs.py
- CSV logs saved to results/
- Figures saved to figures/
