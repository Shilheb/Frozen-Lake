import os

import pandas as pd

from configs import (
    AGG_METRICS_FILENAME,
    ALGO_CONFIGS,
    CASE_CONFIGS,
    OUTPUT_DIR,
    RAW_METRICS_FILENAME,
    SEEDS,
)
from metrics import aggregate_with_confidence_intervals
from plotting import plot_metric_with_ci, plot_policy_comparison
from train_eval import train_and_evaluate_case


def extract_greedy_policy(model, desc):
    n_rows = len(desc)
    n_cols = len(desc[0])
    policy = {}

    for r in range(n_rows):
        for c in range(n_cols):
            state = r * n_cols + c
            cell = desc[r][c]

            if cell in {"S", "F"}:
                action, _ = model.predict(state, deterministic=True)
                action = int(action.item()) if hasattr(action, "item") else int(action)
                policy[state] = action

    return policy


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    figures_dir = os.path.join(OUTPUT_DIR, "figures")
    policy_dir = os.path.join(OUTPUT_DIR, "policies")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(policy_dir, exist_ok=True)

    all_runs = []
    final_models = {}

    for case_cfg in CASE_CONFIGS:
        for algo_cfg in ALGO_CONFIGS:
            for seed in SEEDS:
                print(f"[RUN] case={case_cfg.name} | algo={algo_cfg.name} | seed={seed}")
                df_run, model = train_and_evaluate_case(case_cfg, algo_cfg, seed)
                all_runs.append(df_run)

                # On conserve le modèle de la seed 0 pour visualiser une politique apprise concrète.
                if seed == 0:
                    final_models[(case_cfg.name, algo_cfg.name)] = model

    if not all_runs:
        raise RuntimeError("No runs were executed. Check ALGO_CONFIGS and CASE_CONFIGS.")

    raw_df = pd.concat(all_runs, ignore_index=True)
    raw_csv_path = os.path.join(OUTPUT_DIR, RAW_METRICS_FILENAME)
    raw_df.to_csv(raw_csv_path, index=False)
    print(f"[OK] Raw metrics saved to: {raw_csv_path}")

    agg_df = aggregate_with_confidence_intervals(raw_df)
    agg_csv_path = os.path.join(OUTPUT_DIR, AGG_METRICS_FILENAME)
    agg_df.to_csv(agg_csv_path, index=False)
    print(f"[OK] Aggregated metrics saved to: {agg_csv_path}")

    plot_metric_with_ci(
        agg_df=agg_df,
        metric_base_name="success_rate",
        title="Taux de succès moyen avec IC95",
        y_label="Taux de succès",
        output_path=os.path.join(figures_dir, "success_rate_ci95.png"),
    )

    plot_metric_with_ci(
        agg_df=agg_df,
        metric_base_name="avg_return",
        title="Retour moyen avec IC95",
        y_label="Retour moyen",
        output_path=os.path.join(figures_dir, "avg_return_ci95.png"),
    )

    plot_metric_with_ci(
        agg_df=agg_df,
        metric_base_name="hole_rate",
        title="Taux de chute moyen avec IC95",
        y_label="Taux de chute dans les trous",
        output_path=os.path.join(figures_dir, "hole_rate_ci95.png"),
    )

    plot_metric_with_ci(
        agg_df=agg_df,
        metric_base_name="avg_episode_length",
        title="Longueur moyenne des épisodes avec IC95",
        y_label="Longueur moyenne des épisodes",
        output_path=os.path.join(figures_dir, "avg_episode_length_ci95.png"),
    )

    for case_cfg in CASE_CONFIGS:
        policies = {}
        for algo_cfg in ALGO_CONFIGS:
            key = (case_cfg.name, algo_cfg.name)
            if key in final_models:
                policies[algo_cfg.name] = extract_greedy_policy(final_models[key], case_cfg.desc)

        plot_policy_comparison(
            case_name=case_cfg.name,
            desc=case_cfg.desc,
            policies=policies,
            output_path=os.path.join(policy_dir, f"{case_cfg.name}_policy_comparison.png"),
        )

    print("[DONE] All experiments, aggregation, plots, and policy visualizations completed.")


if __name__ == "__main__":
    main()
