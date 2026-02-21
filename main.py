from pathlib import Path
import pandas as pd
import logging

from src.data_loader import load_ratings
from src.train_test_split import temporal_train_test_split

from src.user_based_cf import train_user_based_model, get_all_predictions as user_predictions
from src.item_based_cf import train_item_based_model, get_all_predictions as item_predictions
from src.matrix_factorization import train_svd_model, get_all_predictions as svd_predictions

from src.evaluator import calculate_rmse, calculate_mae, precision_at_k, recall_at_k

from src.recommender import recommend_top_n
from src.cold_start import get_recommendations_with_cold_start
from src.visualization import plot_model_comparison
from src.embedding_visualization import visualize_item_embeddings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(name, predictions_df):

    return {
        "Model": name,
        "RMSE": calculate_rmse(predictions_df),
        "MAE": calculate_mae(predictions_df),
        "Precision@10": precision_at_k(predictions_df, k=10),
        "Recall@10": recall_at_k(predictions_df, k=10)
    }


def main():

    logger.info("Loading dataset")

    data_path = Path("data/ml-100k")
    ratings_df = load_ratings(str(data_path))

    logger.info("Splitting dataset")

    train_df, test_df = temporal_train_test_split(ratings_df)

    movies_df = ratings_df[['item_id', 'title']].drop_duplicates()

    logger.info("Training User CF")

    user_model = train_user_based_model(train_df)
    user_preds = user_predictions(user_model, test_df)

    logger.info("Training Item CF")

    item_model = train_item_based_model(train_df)
    item_preds = item_predictions(item_model, test_df)

    logger.info("Training SVD")

    svd_model = train_svd_model(train_df)
    svd_preds = svd_predictions(svd_model, test_df)

    logger.info("Evaluating models")

    results = [
        evaluate_model("User-Based CF", user_preds),
        evaluate_model("Item-Based CF", item_preds),
        evaluate_model("SVD", svd_preds)
    ]

    results_df = pd.DataFrame(results)

    print(results_df)

    results_df.to_csv("results.csv", index=False)

    plot_model_comparison(results_df)

    visualize_item_embeddings(svd_model, movies_df)

    logger.info("Generating recommendations")

    recs = recommend_top_n(
        svd_model,
        train_df,
        movies_df,
        user_id=1,
        n=10
    )

    print("\nRecommendations for user 1")
    print(recs)

    logger.info("Testing cold-start recommendations")

    cold_recs = get_recommendations_with_cold_start(
        svd_model,
        train_df,
        movies_df,
        user_id=9999,
        n=10
    )

    print("\nCold-start recommendations")
    print(cold_recs)


if __name__ == "__main__":
    main()