from train_eval import run_experiment
from plotting import plot_metric


def main():
    df = run_experiment(output_dir="../results")

    plot_metric(df, "eval_success_rate", "Success rate", save_dir="../figures")
    plot_metric(df, "eval_hole_rate", "Hole rate", save_dir="../figures")
    plot_metric(df, "eval_return_mean", "Average return", save_dir="../figures")

    print("Expériences terminées.")
    print("Résultats sauvegardés dans ../results")
    print("Figures sauvegardées dans ../figures")


if __name__ == "__main__":
    main()