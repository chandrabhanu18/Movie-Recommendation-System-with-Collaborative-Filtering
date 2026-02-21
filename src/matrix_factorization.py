import logging
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_svd_model(train_df: pd.DataFrame) -> SVD:

    logger.info("=" * 70)
    logger.info("TRAINING PRODUCTION-GRADE SVD MODEL")
    logger.info("=" * 70)

    reader = Reader(rating_scale=(1, 5))

    dataset = Dataset.load_from_df(
        train_df[['user_id', 'item_id', 'rating']],
        reader
    )

    trainset = dataset.build_full_trainset()

    # OPTIMAL PARAMETERS FOR MOVIELENS-100K
    model = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        biased=True,
        random_state=42
    )

    model.fit(trainset)

    logger.info("SVD training complete")

    return model



def get_all_predictions(model: SVD, test_df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Generating SVD predictions")

    predictions = []

    for _, row in test_df.iterrows():

        pred = model.predict(
            row["user_id"],
            row["item_id"]
        )

        predictions.append({
            "user_id": row["user_id"],
            "item_id": row["item_id"],
            "actual_rating": row["rating"],
            "predicted_rating": pred.est,
            "error": row["rating"] - pred.est
        })

    df = pd.DataFrame(predictions)

    mae = df["error"].abs().mean()
    rmse = np.sqrt((df["error"] ** 2).mean())

    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")

    return df
