from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _mean_over_seeds(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        df.groupby(["case", "algorithm", "timesteps"], as_index=False)[metric]
        .mean()
        .sort_values("timesteps")
    )


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, save_dir: str = "figures") -> None:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    agg = _mean_over_seeds(df, metric)
    for case in sorted(agg["case"].unique()):
        case_df = agg[agg["case"] == case]
        plt.figure(figsize=(6, 4))
        for algo in sorted(case_df["algorithm"].unique()):
            sub = case_df[case_df["algorithm"] == algo]
            plt.plot(sub["timesteps"], sub[metric], label=algo)
        plt.xlabel("Timesteps")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} - {case}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path / f"{case}_{metric}.png", dpi=200)
        plt.close()
