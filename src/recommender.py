"""
Movie Recommendation Module

This module provides functions for generating personalized movie recommendations
using trained collaborative filtering models. It handles unseen item identification,
prediction generation, and ranking to produce top-N recommendation lists.

RECOMMENDATION WORKFLOW:

1. **Input**: Trained model + user preferences + item catalog
2. **Identify**: Find items the user hasn't rated yet
3. **Predict**: Use model to predict ratings for unseen items
4. **Rank**: Sort predictions by predicted rating (descending)
5. **Filter**: Select top-N items with highest predicted ratings
6. **Output**: Recommendations with titles and predicted ratings

KEY CONCEPTS:

Cold Start Problem:
- New users with few/no ratings: Cannot recommend accurately
- New items with few/no ratings: Cannot be recommended effectively
- This module requires users to have training data for predictions

Filtering Strategy:
- Exclude already-rated items (user has already seen them)
- Focus on unexplored items in the catalog
- Ensures recommendations provide new discovery opportunities

Recommendation Diversity:
- This implementation ranks purely by predicted rating
- Production systems often add diversity constraints
- Consider genre diversity, temporal diversity, popularity bias mitigation

Scalability Considerations:
- For large catalogs (millions of items), predicting all items is expensive
- Production systems use candidate generation + ranking stages
- This implementation is suitable for catalogs up to ~100k items

Example Usage:
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> import data_loader
    >>> import train_test_split
    >>> import user_based_cf
    >>> import recommender
    >>> 
    >>> # Load and prepare data
    >>> data_path = Path("data/ml-100k")
    >>> ratings_df = data_loader.load_ratings(str(data_path))
    >>> train_df, test_df = train_test_split.temporal_train_test_split(ratings_df)
    >>> 
    >>> # Create movies DataFrame
    >>> movies_df = ratings_df[['item_id', 'title']].drop_duplicates()
    >>> 
    >>> # Train model
    >>> model = user_based_cf.train_user_based_model(train_df)
    >>> 
    >>> # Get top-10 recommendations for user 1
    >>> recommendations = recommender.recommend_top_n(
    ...     model=model,
    ...     train_df=train_df,
    ...     movies_df=movies_df,
    ...     user_id=1,
    ...     n=10
    ... )
    >>> 
    >>> print(f"Top 10 recommendations for user 1:")
    >>> print(recommendations)
"""

import logging
from typing import Any, Union
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def recommend_top_n(
    model: Any,
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_id: int,
    n: int = 10
) -> pd.DataFrame:
    """
    Generate top-N movie recommendations for a user using a trained model.
    
    This function identifies movies the user hasn't rated, predicts ratings for them
    using the trained model, and returns the top-N highest-rated predictions with
    movie titles.
    
    Workflow:
        1. Extract all available movie IDs from the catalog
        2. Identify movies the user has already rated (from training data)
        3. Filter to unseen movies (not rated by user)
        4. Use model to predict ratings for all unseen movies
        5. Sort predictions by predicted rating (descending)
        6. Select top-N highest predicted ratings
        7. Merge with movie titles for human-readable output
    
    Args:
        model (Any): Trained recommendation model with a predict() method.
                    Compatible with Surprise library models (SVD, KNNBasic, etc.).
                    Model must support: model.predict(user_id, item_id)
        
        train_df (pd.DataFrame): Training data used to train the model.
                                Contains columns: 'user_id', 'item_id', 'rating'
                                Used to identify which movies user has already rated.
        
        movies_df (pd.DataFrame): Movie catalog DataFrame.
                                 Must contain columns: 'item_id', 'title'
                                 Provides movie titles for recommendations.
        
        user_id (int): The user for whom to generate recommendations.
                      Must be a user present in the training data.
        
        n (int): Number of top recommendations to return.
                Default: 10
                Typical values: 5, 10, 20, 50
    
    Returns:
        pd.DataFrame: DataFrame with top-N recommendations containing columns:
                     'item_id': Movie identifier
                     'title': Movie title  
                     'predicted_rating': Model's predicted rating for this movie
                     
                     Sorted by predicted_rating in descending order.
    
    Raises:
        ValueError: If inputs are invalid (empty DataFrames, invalid n, missing columns).
        TypeError: If model doesn't have predict() method.
        KeyError: If required columns are missing from DataFrames.
        Exception: For other unexpected errors during recommendation generation.
    
    Notes:
        - If user has rated all movies, returns empty DataFrame
        - If user is not in training data, may use global mean for predictions
        - Predictions are bounded by the rating scale (e.g., 1-5)
        - For new users (cold start), predictions may be inaccurate
        - Computational cost: O(M) where M = number of unseen movies
    
    Example:
        >>> # Train model
        >>> model = user_based_cf.train_user_based_model(train_df)
        >>> 
        >>> # Get top-10 recommendations
        >>> recommendations = recommend_top_n(
        ...     model=model,
        ...     train_df=train_df,
        ...     movies_df=movies_df,
        ...     user_id=1,
        ...     n=10
        ... )
        >>> 
        >>> print("Top 10 Recommendations:")
        >>> for idx, row in recommendations.iterrows():
        ...     print(f"{idx+1}. {row['title']}: {row['predicted_rating']:.2f}")
        >>> 
        >>> # Get top-5 recommendations for different user
        >>> recommendations_user_2 = recommend_top_n(
        ...     model=model,
        ...     train_df=train_df,
        ...     movies_df=movies_df,
        ...     user_id=2,
        ...     n=5
        ... )
    """
    try:
        # Validate model
        if model is None:
            raise ValueError("Model is None. Please provide a trained model.")
        
        if not hasattr(model, 'predict'):
            raise TypeError("Model must have a predict() method")
        
        # Validate train_df
        if train_df.empty:
            raise ValueError("Training DataFrame is empty")
        
        required_train_columns = ['user_id', 'item_id', 'rating']
        missing_train_cols = [col for col in required_train_columns 
                             if col not in train_df.columns]
        if missing_train_cols:
            raise ValueError(f"train_df missing required columns: {missing_train_cols}")
        
        # Validate movies_df
        if movies_df.empty:
            raise ValueError("Movies DataFrame is empty")
        
        required_movie_columns = ['item_id', 'title']
        missing_movie_cols = [col for col in required_movie_columns 
                             if col not in movies_df.columns]
        if missing_movie_cols:
            raise ValueError(f"movies_df missing required columns: {missing_movie_cols}")
        
        # Validate parameters
        if not isinstance(user_id, (int, np.integer)):
            raise TypeError(f"user_id must be an integer, got {type(user_id)}")
        
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")
        
        logger.info("=" * 70)
        logger.info(f"GENERATING TOP-{n} RECOMMENDATIONS FOR USER {user_id}")
        logger.info("=" * 70)
        
        # Step 1: Get all available movie IDs from catalog
        all_movie_ids = set(movies_df['item_id'].unique())
        logger.info(f"Total movies in catalog: {len(all_movie_ids)}")
        
        # Step 2: Get movies already rated by this user
        user_rated_movies = set(train_df[train_df['user_id'] == user_id]['item_id'].unique())
        logger.info(f"Movies already rated by user {user_id}: {len(user_rated_movies)}")
        
        # Step 3: Find unseen movies (movies not rated by user)
        unseen_movies = all_movie_ids - user_rated_movies
        unseen_movies = sorted(list(unseen_movies))
        logger.info(f"Unseen movies to predict: {len(unseen_movies)}")
        
        if len(unseen_movies) == 0:
            logger.warning(f"User {user_id} has rated all movies. No recommendations available.")
            return pd.DataFrame(columns=['item_id', 'title', 'predicted_rating'])
        
        # Step 4: Predict ratings for all unseen movies
        logger.info(f"Generating predictions for {len(unseen_movies)} unseen movies...")
        
        predictions_list = []
        failed_predictions = 0
        
        for item_id in unseen_movies:
            try:
                # Use model.predict() - compatible with Surprise library
                prediction = model.predict(user_id, item_id, verbose=False)
                predicted_rating = prediction.est
                
                predictions_list.append({
                    'item_id': item_id,
                    'predicted_rating': predicted_rating
                })
            except Exception as e:
                failed_predictions += 1
                logger.debug(f"Failed to predict for item {item_id}: {e}")
                continue
        
        if failed_predictions > 0:
            logger.warning(f"Failed to generate {failed_predictions} predictions")
        
        if not predictions_list:
            raise ValueError(f"Could not generate any predictions for user {user_id}")
        
        # Convert to DataFrame
        predictions_df = pd.DataFrame(predictions_list)
        logger.info(f"Successfully generated {len(predictions_df)} predictions")
        
        # Step 5: Sort by predicted rating (descending)
        predictions_df = predictions_df.sort_values(
            'predicted_rating', 
            ascending=False
        ).reset_index(drop=True)
        
        # Step 6: Select top-N recommendations
        top_n_predictions = predictions_df.head(n)
        logger.info(f"Selected top-{n} recommendations")
        
        # Step 7: Merge with movie titles
        recommendations = top_n_predictions.merge(
            movies_df[['item_id', 'title']],
            on='item_id',
            how='left'
        )
        
        # Reorder columns for clarity
        recommendations = recommendations[['item_id', 'title', 'predicted_rating']]
        
        # Log statistics
        avg_predicted_rating = recommendations['predicted_rating'].mean()
        min_predicted_rating = recommendations['predicted_rating'].min()
        max_predicted_rating = recommendations['predicted_rating'].max()
        
        logger.info(f"Recommendation statistics:")
        logger.info(f"  Average predicted rating: {avg_predicted_rating:.2f}")
        logger.info(f"  Rating range: [{min_predicted_rating:.2f}, {max_predicted_rating:.2f}]")
        logger.info("=" * 70)
        
        return recommendations
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except TypeError as e:
        logger.error(f"Type error: {e}")
        raise
    except KeyError as e:
        logger.error(f"Key error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating recommendations: {e}")
        raise


if __name__ == "__main__":
    """
    Example usage and testing of the recommender module.
    """
    try:
        from pathlib import Path
        import sys
        
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        import data_loader
        import train_test_split
        import user_based_cf
        
        print("\n" + "=" * 70)
        print("MOVIE RECOMMENDATION SYSTEM - EXAMPLE USAGE")
        print("=" * 70 + "\n")
        
        # Load data
        data_path = Path(__file__).parent.parent / "data" / "ml-100k"
        
        if not data_path.exists():
            print(f"❌ Dataset not found at {data_path}")
            print("Please ensure the MovieLens 100K dataset is located at data/ml-100k/")
            sys.exit(1)
        
        print("1. Loading ratings data...")
        ratings_df = data_loader.load_ratings(str(data_path))
        print(f"   ✓ Loaded {len(ratings_df)} ratings\n")
        
        print("2. Performing temporal train-test split...")
        train_df, test_df = train_test_split.temporal_train_test_split(
            ratings_df,
            test_size=0.2
        )
        print(f"   ✓ Train set: {len(train_df)} ratings")
        print(f"   ✓ Test set: {len(test_df)} ratings\n")
        
        print("3. Preparing movies catalog...")
        movies_df = ratings_df[['item_id', 'title']].drop_duplicates().reset_index(drop=True)
        print(f"   ✓ Catalog contains {len(movies_df)} movies\n")
        
        print("4. Training user-based CF model...")
        # Use subset for faster training in example
        train_subset = train_df.head(5000)
        model = user_based_cf.train_user_based_model(train_subset)
        print(f"   ✓ Model trained\n")
        
        print("5. Generating recommendations for sample users...")
        
        # Get recommendations for multiple users
        sample_users = train_subset['user_id'].unique()[:3]
        
        for user_id in sample_users:
            print(f"\n   User {user_id} - Top 5 Recommendations:")
            print("   " + "-" * 66)
            
            recommendations = recommend_top_n(
                model=model,
                train_df=train_subset,
                movies_df=movies_df,
                user_id=int(user_id),
                n=5
            )
            
            for idx, row in recommendations.iterrows():
                print(f"   {idx+1}. {row['title']:<50} | Rating: {row['predicted_rating']:.2f}")
        
        print("\n" + "=" * 70)
        print("✓ Recommendation generation completed successfully!")
        print("=" * 70 + "\n")
        
        # Demonstrate different N values
        print("6. Comparing different N values for User 1:")
        test_user = int(sample_users[0])
        
        for n_value in [3, 5, 10]:
            recs = recommend_top_n(
                model=model,
                train_df=train_subset,
                movies_df=movies_df,
                user_id=test_user,
                n=n_value
            )
            avg_rating = recs['predicted_rating'].mean()
            print(f"   Top-{n_value}: Average predicted rating = {avg_rating:.2f}")
        
        print("\n" + "=" * 70 + "\n")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are in the same directory")
        logger.exception("Failed during example usage")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Failed to run example usage")
