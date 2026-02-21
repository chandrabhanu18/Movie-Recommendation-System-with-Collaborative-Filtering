"""
Cold Start Handling Module
Production-grade fallback recommendation strategies.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_popular_movies(
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    n: int = 10,
    min_ratings: int = 50
) -> pd.DataFrame:
    """
    Return most popular movies based on rating count and average rating.
    """

    logger.info("Generating popular movie recommendations (cold-start)")

    stats = (
        train_df
        .groupby("item_id")
        .agg(
            rating_count=("rating", "count"),
            avg_rating=("rating", "mean")
        )
        .reset_index()
    )

    # filter movies with enough ratings
    stats = stats[stats["rating_count"] >= min_ratings]

    # weighted score
    stats["score"] = stats["avg_rating"] * stats["rating_count"]

    stats = stats.sort_values("score", ascending=False)

    recommendations = stats.head(n).merge(
        movies_df,
        on="item_id",
        how="left"
    )

    return recommendations[["item_id", "title", "avg_rating", "rating_count"]]


def get_recommendations_with_cold_start(
    model,
    train_df,
    movies_df,
    user_id,
    n=10
):
    """
    Hybrid recommendation function with cold-start fallback.
    """

    user_exists = user_id in train_df["user_id"].values

    if not user_exists:
        logger.info(f"Cold start detected for user {user_id}")
        return get_popular_movies(train_df, movies_df, n)

    from src.recommender import recommend_top_n

    return recommend_top_n(
        model=model,
        train_df=train_df,
        movies_df=movies_df,
        user_id=user_id,
        n=n
    )