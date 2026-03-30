import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ARROW_MAP = {
    0: "←",  # LEFT
    1: "↓",  # DOWN
    2: "→",  # RIGHT
    3: "↑",  # UP
}


def plot_metric_with_ci(
    agg_df: pd.DataFrame,
    metric_base_name: str,
    title: str,
    y_label: str,
    output_path: str,
) -> None:
    cases = agg_df["case"].unique().tolist()

    fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 4), squeeze=False)
    axes = axes[0]

    for ax, case_name in zip(axes, cases):
        subset_case = agg_df[agg_df["case"] == case_name]

        for algo_name in subset_case["algo"].unique():
            subset = subset_case[subset_case["algo"] == algo_name].sort_values("timestep")

            x = subset["timestep"].to_numpy()
            y = subset[f"{metric_base_name}_mean"].to_numpy()
            ci = subset[f"{metric_base_name}_ci95"].to_numpy()

            ax.plot(x, y, label=algo_name)
            ax.fill_between(x, y - ci, y + ci, alpha=0.2)

        ax.set_title(case_name)
        ax.set_xlabel("Pas de temps d'entraînement")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _draw_grid_policy(ax, desc, policy_dict, title: str) -> None:
    n_rows = len(desc)
    n_cols = len(desc[0])

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, n_cols + 1, 1))
    ax.set_yticks(np.arange(0, n_rows + 1, 1))
    ax.grid(True)

    for r in range(n_rows):
        for c in range(n_cols):
            cell = desc[r][c]

            if cell == "H":
                facecolor = "#d9d9d9"
            elif cell == "S":
                facecolor = "#dff0d8"
            elif cell == "G":
                facecolor = "#fce5cd"
            else:
                facecolor = "white"

            rect = plt.Rectangle((c, r), 1, 1, fill=True, alpha=0.5, edgecolor="black", facecolor=facecolor)
            ax.add_patch(rect)

            state_idx = r * n_cols + c

            if cell in {"S", "F"} and state_idx in policy_dict:
                ax.text(
                    c + 0.5,
                    r + 0.55,
                    ARROW_MAP.get(policy_dict[state_idx], "?"),
                    ha="center",
                    va="center",
                    fontsize=16,
                )
            else:
                ax.text(
                    c + 0.5,
                    r + 0.55,
                    cell,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

    ax.set_title(title)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def plot_policy_comparison(case_name: str, desc, dqn_policy, ppo_policy, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    _draw_grid_policy(axes[0], desc, dqn_policy, f"{case_name} — Politique DQN")
    _draw_grid_policy(axes[1], desc, ppo_policy, f"{case_name} — Politique PPO")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)