import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_comparison(results_df, save_path="model_comparison.png"):
    """
    Create visualization comparing model performance

    Parameters:
    results_df : DataFrame with columns:
        Model, RMSE, MAE, Precision@10, Recall@10

    save_path : output image path
    """

    sns.set(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # RMSE
    sns.barplot(
        x="Model",
        y="RMSE",
        data=results_df,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title("RMSE Comparison")

    # MAE
    sns.barplot(
        x="Model",
        y="MAE",
        data=results_df,
        ax=axes[0, 1]
    )
    axes[0, 1].set_title("MAE Comparison")

    # Precision
    sns.barplot(
        x="Model",
        y="Precision@10",
        data=results_df,
        ax=axes[1, 0]
    )
    axes[1, 0].set_title("Precision@10 Comparison")

    # Recall
    sns.barplot(
        x="Model",
        y="Recall@10",
        data=results_df,
        ax=axes[1, 1]
    )
    axes[1, 1].set_title("Recall@10 Comparison")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Visualization saved to {save_path}")