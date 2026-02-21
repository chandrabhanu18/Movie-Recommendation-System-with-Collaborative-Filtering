import logging
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNBasic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_K_NEIGHBORS = 40
DEFAULT_MIN_K = 1


def train_item_based_model(train_df: pd.DataFrame) -> KNNBasic:
    """
    Train Item-Based Collaborative Filtering model.
    """

    if train_df.empty:
        raise ValueError("Training DataFrame is empty")

    required_columns = ["user_id", "item_id", "rating"]
    missing = [c for c in required_columns if c not in train_df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ---------- Stats ----------
    n_users = train_df["user_id"].nunique()
    n_items = train_df["item_id"].nunique()
    n_ratings = len(train_df)

    # FIXED sparsity calculation
    sparsity = (1 - (n_ratings / (n_users * n_items))) * 100

    logger.info("=" * 70)
    logger.info("TRAINING ITEM-BASED CF MODEL")
    logger.info("=" * 70)

    logger.info(f"Training shape: {train_df.shape}")
    logger.info(f"Users: {n_users}")
    logger.info(f"Items: {n_items}")
    logger.info(f"Ratings: {n_ratings}")
    logger.info(f"Sparsity: {sparsity:.2f}%")

    rating_min = float(train_df["rating"].min())
    rating_max = float(train_df["rating"].max())

    logger.info(f"Rating scale: {rating_min} to {rating_max}")

    # ---------- Surprise Dataset ----------
    reader = Reader(rating_scale=(rating_min, rating_max))

    dataset = Dataset.load_from_df(
        train_df[["user_id", "item_id", "rating"]],
        reader
    )

    trainset = dataset.build_full_trainset()

    logger.info("Trainset built")
    logger.info(f"Users: {trainset.n_users}")
    logger.info(f"Items: {trainset.n_items}")
    logger.info(f"Ratings: {trainset.n_ratings}")

    # ---------- Model ----------
    sim_options = {
        "name": "cosine",
        "user_based": False,   # IMPORTANT difference
        "min_support": 1
    }

    logger.info("Training item-based KNNBasic model")

    model = KNNBasic(
        k=DEFAULT_K_NEIGHBORS,
        min_k=DEFAULT_MIN_K,
        sim_options=sim_options,
        verbose=False
    )

    model.fit(trainset)

    logger.info("Training complete")
    logger.info("=" * 70)

    return model


def predict_rating(
    model: KNNBasic,
    user_id: int,
    item_id: int
) -> float:

    if model is None:
        raise ValueError("Model is None")

    prediction = model.predict(user_id, item_id)

    return float(prediction.est)


def get_all_predictions(
    model: KNNBasic,
    test_df: pd.DataFrame
) -> pd.DataFrame:

    if test_df.empty:
        raise ValueError("Test DataFrame is empty")

    logger.info("=" * 70)
    logger.info("GENERATING ITEM-BASED PREDICTIONS")
    logger.info("=" * 70)

    predictions = []

    for _, row in test_df.iterrows():

        user_id = int(row["user_id"])
        item_id = int(row["item_id"])
        actual = float(row["rating"])

        try:

            predicted = predict_rating(model, user_id, item_id)

            predictions.append({
                "user_id": user_id,
                "item_id": item_id,
                "actual_rating": actual,
                "predicted_rating": predicted,
                "error": actual - predicted
            })

        except Exception as e:

            logger.warning(
                f"Prediction failed for user {user_id}, item {item_id}: {e}"
            )

    predictions_df = pd.DataFrame(predictions)

    mae = predictions_df["error"].abs().mean()
    rmse = np.sqrt((predictions_df["error"] ** 2).mean())

    logger.info(f"Predictions: {len(predictions_df)}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")

    within_1 = (predictions_df["error"].abs() <= 1).mean() * 100
    within_half = (predictions_df["error"].abs() <= 0.5).mean() * 100

    logger.info(f"Within ±0.5: {within_half:.1f}%")
    logger.info(f"Within ±1.0: {within_1:.1f}%")

    logger.info("=" * 70)

    return predictions_df
 