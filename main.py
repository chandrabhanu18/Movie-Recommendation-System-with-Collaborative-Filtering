from pathlib import Path
import pandas as pd

from src.data_loader import load_ratings
from src.train_test_split import temporal_train_test_split

from src.user_based_cf import (
    train_user_based_model,
    get_all_predictions as user_predictions
)

from src.item_based_cf import (
    train_item_based_model,
    get_all_predictions as item_predictions
)

from src.matrix_factorization import (
    train_svd_model,
    get_all_predictions as svd_predictions
)

from src.evaluator import (
    calculate_rmse,
    calculate_mae,
    precision_at_k,
    recall_at_k
)

from src.recommender import recommend_top_n
from src.visualization import plot_model_comparison


def evaluate_model(name, predictions_df):
    """Evaluate model performance and return metrics dictionary"""

    rmse = calculate_rmse(predictions_df)
    mae = calculate_mae(predictions_df)
    precision = precision_at_k(predictions_df, k=10)
    recall = recall_at_k(predictions_df, k=10)

    return {
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "Precision@10": precision,
        "Recall@10": recall
    }


def main():

    print("\n" + "=" * 60)
    print("MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    data_path = Path("data/ml-100k")
    ratings_df = load_ratings(str(data_path))

    # Split dataset
    print("\nSplitting dataset...")
    train_df, test_df = temporal_train_test_split(ratings_df)

    movies_df = ratings_df[['item_id', 'title']].drop_duplicates()

    # Train User-Based CF
    print("\nTraining User-Based CF...")
    user_model = train_user_based_model(train_df)
    user_preds = user_predictions(user_model, test_df)

    # Train Item-Based CF
    print("\nTraining Item-Based CF...")
    item_model = train_item_based_model(train_df)
    item_preds = item_predictions(item_model, test_df)

    # Train SVD
    print("\nTraining SVD...")
    svd_model = train_svd_model(train_df)
    svd_preds = svd_predictions(svd_model, test_df)

    # Evaluate models
    print("\nEvaluating models...")

    results = []

    results.append(evaluate_model("User-Based CF", user_preds))
    results.append(evaluate_model("Item-Based CF", item_preds))
    results.append(evaluate_model("SVD", svd_preds))

    results_df = pd.DataFrame(results)

    print("\nMODEL COMPARISON:")
    print(results_df)

    # Save results
    results_df.to_csv("results.csv", index=False)
    print("\nSaved results.csv")

    # Generate visualization
    plot_model_comparison(results_df)
    print("Saved model_comparison.png")

    # Generate recommendations
    print("\nGenerating recommendations for user 1:")

    recs = recommend_top_n(
        model=svd_model,
        train_df=train_df,
        movies_df=movies_df,
        user_id=1,
        n=10
    )

    print(recs)

    # Save recommendations
    recs.to_csv("recommendations_user_1.csv", index=False)
    print("\nSaved recommendations_user_1.csv")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()