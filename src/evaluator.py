"""
Recommender System Evaluation Module

This module provides comprehensive evaluation metrics for recommender systems.
It includes both rating prediction metrics (RMSE, MAE) and ranking metrics
(Precision@K, Recall@K).

EVALUATION METRICS OVERVIEW:

1. RATING PREDICTION METRICS:
   
   These metrics evaluate how well the model predicts actual rating values.
   
   - RMSE (Root Mean Squared Error):
     * Measures average prediction error in rating scale units
     * Penalizes large errors more than small errors (quadratic penalty)
     * Sensitive to outliers
     * Range: [0, ∞]
     * Lower is better
     * Example: RMSE=0.5 means predictions are off by ~0.5 stars on average
   
   - MAE (Mean Absolute Error):
     * Measures average absolute prediction error
     * Treats all errors equally (linear penalty)
     * More robust to outliers than RMSE
     * Range: [0, ∞]
     * Lower is better
     * Example: MAE=0.5 means predictions are off by 0.5 stars on average

2. RANKING METRICS (Top-K):
   
   These metrics evaluate how well the model ranks items (which to recommend).
   They treat items above a rating threshold as "relevant" and evaluate if the
   model's top-K recommendations include relevant items.
   
   - Precision@K:
     * What fraction of the top-K recommendations are relevant?
     * Precision@K = (# relevant items in top-K) / K
     * Answers: "Of the K items I recommended, how many were any good?"
     * Range: [0, 1]
     * Higher is better
     * Example: If we recommend 10 items and 7 are good, Precision@10 = 0.7
   
   - Recall@K:
     * Of all relevant items, what fraction appears in top-K?
     * Recall@K = (# relevant items in top-K) / (# total relevant items)
     * Answers: "Did I catch all the good items in my top-K recommendations?"
     * Range: [0, 1]
     * Higher is better
     * Example: If user has 20 good movies and we recommend 7 of them in top-10,
               Recall@10 = 7/20 = 0.35
   
   - Relationship between metrics:
     * Precision focuses on recommendation quality (avoiding bad recommendations)
     * Recall focuses on recommendation coverage (finding all good items)
     * Trade-off: high precision often means low recall, and vice versa
     * Often need to balance (Precision-Recall curve)

WHY DIFFERENT METRICS:

Different use cases need different metrics:

1. **Music Streaming (High Precision Priority)**:
   - Users are impatient; bad recommendations = quit the app
   - Care more about avoiding bad recommendations (precision)
   - 3-5 recommendations, high quality preferred
   
2. **E-Commerce (Balanced Approach)**:
   - Users browse multiple items
   - Need both quality (precision) and coverage (recall)
   - 10-20 recommendations is typical
   
3. **News/Content Discovery (Higher Recall Priority)**:
   - Users want to explore broadly
   - Prefer seeing diverse items, even if some are less relevant
   - 20-50+ recommendations acceptable

WHEN TO USE WHICH METRIC:

- **RMSE/MAE**: 
  * When you care about predicting exact ratings
  * Rating prediction systems (explicit feedback: 1-5 stars)
  * Example: "Predict what rating this user will give this movie"

- **Precision@K**:
  * When false positives are expensive
  * Limited recommendation slots (top 3-5 items)
  * Example: "Show only the best recommendations"

- **Recall@K**:
  * When missing relevant items is costly
  * Browsing scenarios where many items are relevant
  * Example: "Help me find all movies I'll like"

Example Usage:
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> import data_loader
    >>> import train_test_split
    >>> import user_based_cf
    >>> import evaluator
    >>> 
    >>> # Load and prepare data
    >>> data_path = Path("data/ml-100k")
    >>> ratings_df = data_loader.load_ratings(str(data_path))
    >>> train_df, test_df = train_test_split.temporal_train_test_split(ratings_df)
    >>> 
    >>> # Train model and get predictions
    >>> model = user_based_cf.train_user_based_model(train_df)
    >>> predictions_df = user_based_cf.get_all_predictions(model, test_df)
    >>> 
    >>> # Evaluate model
    >>> rmse = evaluator.calculate_rmse(predictions_df)
    >>> mae = evaluator.calculate_mae(predictions_df)
    >>> precision_at_10 = evaluator.precision_at_k(predictions_df, k=10)
    >>> recall_at_10 = evaluator.recall_at_k(predictions_df, k=10)
    >>> 
    >>> print(f"RMSE: {rmse:.4f}")
    >>> print(f"MAE: {mae:.4f}")
    >>> print(f"Precision@10: {precision_at_10:.4f}")
    >>> print(f"Recall@10: {recall_at_10:.4f}")
"""

import logging
from typing import Union
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_rmse(predictions_df: pd.DataFrame) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) for rating predictions.
    
    RMSE measures the average magnitude of prediction errors. It heavily penalizes
    large errors due to the squaring operation, making it sensitive to outliers.
    
    Formula:
        RMSE = sqrt((1/N) * Σ(actual_rating - predicted_rating)²)
    
    Where:
        - N = number of predictions
        - actual_rating = true rating from test set
        - predicted_rating = model's predicted rating
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with columns:
                                      'actual_rating': Ground truth ratings
                                      'predicted_rating': Model predictions
    
    Returns:
        float: RMSE value (typically in same scale as ratings, e.g., 0-5).
              Value of 0.5 means predictions are off by 0.5 stars on average.
    
    Raises:
        ValueError: If DataFrame is empty or required columns are missing.
        TypeError: If rating columns contain non-numeric values.
        Exception: For other unexpected errors.
    
    Notes:
        - Lower values are better (0 = perfect predictions)
        - More sensitive to outliers than MAE due to squaring
        - Useful when large errors are particularly undesirable
        - Common in rating prediction tasks
    
    Example:
        >>> predictions = pd.DataFrame({
        ...     'actual_rating': [5, 4, 3, 2],
        ...     'predicted_rating': [4.5, 4.2, 2.8, 2.3]
        ... })
        >>> rmse = calculate_rmse(predictions)
        >>> print(f"RMSE: {rmse:.4f}")  # Output: RMSE: 0.3162
    """
    try:
        # Validate input
        if predictions_df.empty:
            raise ValueError("Predictions DataFrame is empty")
        
        required_columns = ['actual_rating', 'predicted_rating']
        missing_columns = [col for col in required_columns if col not in predictions_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types
        if not (pd.api.types.is_numeric_dtype(predictions_df['actual_rating']) and
                pd.api.types.is_numeric_dtype(predictions_df['predicted_rating'])):
            raise TypeError("Rating columns must be numeric")
        
        # Extract ratings
        actual = predictions_df['actual_rating'].values
        predicted = predictions_df['predicted_rating'].values
        
        # Check for NaN values
        valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
        
        if not valid_mask.any():
            raise ValueError("No valid rating pairs found (all contain NaN)")
        
        actual_valid = actual[valid_mask]
        predicted_valid = predicted[valid_mask]
        
        # Calculate RMSE
        squared_errors = (actual_valid - predicted_valid) ** 2
        mse = np.mean(squared_errors)
        rmse = np.sqrt(mse)
        
        logger.info(f"RMSE calculated: {rmse:.4f} (based on {len(actual_valid)} predictions)")
        
        return float(rmse)
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except TypeError as e:
        logger.error(f"Type error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calculating RMSE: {e}")
        raise


def calculate_mae(predictions_df: pd.DataFrame) -> float:
    """
    Calculate Mean Absolute Error (MAE) for rating predictions.
    
    MAE measures the average magnitude of prediction errors using absolute values.
    It treats all errors equally, making it more robust to outliers than RMSE.
    
    Formula:
        MAE = (1/N) * Σ|actual_rating - predicted_rating|
    
    Where:
        - N = number of predictions
        - actual_rating = true rating from test set
        - predicted_rating = model's predicted rating
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with columns:
                                      'actual_rating': Ground truth ratings
                                      'predicted_rating': Model predictions
    
    Returns:
        float: MAE value (in same scale as ratings, e.g., 0-5).
              Value of 0.5 means predictions are off by 0.5 stars on average.
    
    Raises:
        ValueError: If DataFrame is empty or required columns are missing.
        TypeError: If rating columns contain non-numeric values.
        Exception: For other unexpected errors.
    
    Notes:
        - Lower values are better (0 = perfect predictions)
        - More interpretable than RMSE (same scale as rating errors)
        - More robust to outliers than RMSE
        - Preferred when all error magnitudes matter equally
        - Often slightly higher than RMSE for same data
    
    Example:
        >>> predictions = pd.DataFrame({
        ...     'actual_rating': [5, 4, 3, 2],
        ...     'predicted_rating': [4.5, 4.2, 2.8, 2.3]
        ... })
        >>> mae = calculate_mae(predictions)
        >>> print(f"MAE: {mae:.4f}")  # Output: MAE: 0.2500
    """
    try:
        # Validate input
        if predictions_df.empty:
            raise ValueError("Predictions DataFrame is empty")
        
        required_columns = ['actual_rating', 'predicted_rating']
        missing_columns = [col for col in required_columns if col not in predictions_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate data types
        if not (pd.api.types.is_numeric_dtype(predictions_df['actual_rating']) and
                pd.api.types.is_numeric_dtype(predictions_df['predicted_rating'])):
            raise TypeError("Rating columns must be numeric")
        
        # Extract ratings
        actual = predictions_df['actual_rating'].values
        predicted = predictions_df['predicted_rating'].values
        
        # Check for NaN values
        valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
        
        if not valid_mask.any():
            raise ValueError("No valid rating pairs found (all contain NaN)")
        
        actual_valid = actual[valid_mask]
        predicted_valid = predicted[valid_mask]
        
        # Calculate MAE
        absolute_errors = np.abs(actual_valid - predicted_valid)
        mae = np.mean(absolute_errors)
        
        logger.info(f"MAE calculated: {mae:.4f} (based on {len(actual_valid)} predictions)")
        
        return float(mae)
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except TypeError as e:
        logger.error(f"Type error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calculating MAE: {e}")
        raise


def precision_at_k(
    predictions_df: pd.DataFrame,
    k: int = 10,
    threshold: float = 4.0
) -> float:
    """
    Calculate Precision@K: What fraction of top-K recommendations are relevant?
    
    For each user, identify the top-K items by predicted rating and check how many
    of those items have actual ratings >= threshold (considered "relevant").
    
    Precision@K = (# relevant items in top-K recommendations) / K
    
    This metric answers: "Of the K items I recommended to each user, how many
    were items they actually liked?"
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with columns:
                                      'user_id': User identifier
                                      'actual_rating': Ground truth rating
                                      'predicted_rating': Model's predicted rating
        
        k (int): Number of top items to consider per user.
                Default: 10 (evaluate top-10 recommendations)
                Common values: 5, 10, 20
        
        threshold (float): Rating value that defines "relevant" items.
                          actual_rating >= threshold = relevant
                          Default: 4.0 (on 1-5 scale)
                          Common values: 3.0, 3.5, 4.0, 4.5
    
    Returns:
        float: Precision@K averaged across all users.
              Range: [0, 1]
              0 = no good recommendations, 1 = all recommendations were good
    
    Raises:
        ValueError: If DataFrame is empty or required columns are missing.
        TypeError: If data types are invalid.
        Exception: For other unexpected errors.
    
    Notes:
        - Calculated per-user, then averaged across users
        - If a user has < K items, use available items
        - Users with no relevant items still participate (precision = 0)
        - K should typically be <= total items to evaluate fairly
        - Cannot directly compare Precision@5 vs Precision@10
    
    Example:
        >>> predictions = pd.DataFrame({
        ...     'user_id': [1, 1, 1, 1, 2, 2, 2, 2],
        ...     'actual_rating': [5, 4, 3, 2, 5, 5, 2, 1],
        ...     'predicted_rating': [4.8, 4.5, 3.2, 1.9, 4.9, 4.7, 2.1, 1.1]
        ... })
        >>> precision_at_5 = precision_at_k(predictions, k=5, threshold=4.0)
        >>> # User 1: Recommended [item0, item1, item2, item3] (4 total)
        >>> #         Relevant in recommendations: items 0,1 (2)
        >>> #         Precision = 2/4 = 0.5
        >>> # User 2: Recommended [item0, item1] (2 total, < k)
        >>> #         Relevant in recommendations: both (2)
        >>> #         Precision = 2/2 = 1.0
        >>> # Overall: (0.5 + 1.0) / 2 = 0.75
    """
    try:
        # Validate input
        if predictions_df.empty:
            raise ValueError("Predictions DataFrame is empty")
        
        required_columns = ['user_id', 'actual_rating', 'predicted_rating']
        missing_columns = [col for col in required_columns if col not in predictions_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"threshold must be numeric, got {type(threshold)}")
        
        logger.info(f"Calculating Precision@{k} (threshold={threshold})")
        
        # Create a working copy
        df = predictions_df.copy()
        
        # Get unique users
        users = df['user_id'].unique()
        logger.info(f"Evaluating {len(users)} users")
        
        precisions = []
        
        for user_id in users:
            user_data = df[df['user_id'] == user_id].copy()
            
            # Sort by predicted rating (descending) and take top-K
            user_data = user_data.sort_values('predicted_rating', ascending=False).head(k)
            
            # Count relevant items (actual_rating >= threshold)
            relevant_items = (user_data['actual_rating'] >= threshold).sum()
            
            # Number of recommendations for this user (might be < k)
            num_recommendations = len(user_data)
            
            # Calculate precision for this user
            if num_recommendations > 0:
                user_precision = relevant_items / num_recommendations
            else:
                user_precision = 0.0
            
            precisions.append(user_precision)
        
        # Average precision across users
        mean_precision = np.mean(precisions) if precisions else 0.0
        
        logger.info(f"Precision@{k}: {mean_precision:.4f}")
        logger.info(f"  Average relevant items per user: {np.mean([p * k for p in precisions]):.2f}")
        
        return float(mean_precision)
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except TypeError as e:
        logger.error(f"Type error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calculating Precision@{k}: {e}")
        raise


def recall_at_k(
    predictions_df: pd.DataFrame,
    k: int = 10,
    threshold: float = 4.0
) -> float:
    """
    Calculate Recall@K: Of all relevant items, what fraction appears in top-K?
    
    For each user, identify all relevant items (actual_rating >= threshold) and
    check how many appear in the top-K recommendations (by predicted rating).
    
    Recall@K = (# relevant items in top-K recommendations) / (# total relevant items)
    
    This metric answers: "Did I find all the good items for this user in my
    top-K recommendations?"
    
    Args:
        predictions_df (pd.DataFrame): DataFrame with columns:
                                      'user_id': User identifier
                                      'actual_rating': Ground truth rating
                                      'predicted_rating': Model's predicted rating
        
        k (int): Number of top items to consider per user.
                Default: 10 (evaluate top-10 recommendations)
                Common values: 5, 10, 20
        
        threshold (float): Rating value that defines "relevant" items.
                          actual_rating >= threshold = relevant
                          Default: 4.0 (on 1-5 scale)
                          Common values: 3.0, 3.5, 4.0, 4.5
    
    Returns:
        float: Recall@K averaged across all users.
              Range: [0, 1]
              0 = missed all good items, 1 = found all good items
    
    Raises:
        ValueError: If DataFrame is empty or required columns are missing.
        TypeError: If data types are invalid.
        Exception: For other unexpected errors.
    
    Notes:
        - Calculated per-user, then averaged across users
        - Users with zero relevant items are excluded from average
          (Recall is undefined for users with no relevant items)
        - If top-K >= total items, recall = 1.0
        - Cannot directly compare Recall@5 vs Recall@10
        - Recall can be high even if recommendations are bad (if K is large)
    
    Example:
        >>> predictions = pd.DataFrame({
        ...     'user_id': [1, 1, 1, 1, 2, 2, 2, 2],
        ...     'actual_rating': [5, 4, 3, 2, 5, 5, 2, 1],
        ...     'predicted_rating': [4.8, 4.5, 3.2, 1.9, 4.9, 4.7, 2.1, 1.1]
        ... })
        >>> recall_at_5 = recall_at_k(predictions, k=5, threshold=4.0)
        >>> # User 1: Relevant items [0, 1] (2 total)
        >>> #         Found in top-K: items 0, 1 (2)
        >>> #         Recall = 2/2 = 1.0
        >>> # User 2: Relevant items [0, 1] (2 total)
        >>> #         Found in top-K: items 0, 1 (2)
        >>> #         Recall = 2/2 = 1.0
        >>> # Overall: (1.0 + 1.0) / 2 = 1.0
    """
    try:
        # Validate input
        if predictions_df.empty:
            raise ValueError("Predictions DataFrame is empty")
        
        required_columns = ['user_id', 'actual_rating', 'predicted_rating']
        missing_columns = [col for col in required_columns if col not in predictions_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        
        if not isinstance(threshold, (int, float)):
            raise TypeError(f"threshold must be numeric, got {type(threshold)}")
        
        logger.info(f"Calculating Recall@{k} (threshold={threshold})")
        
        # Create a working copy
        df = predictions_df.copy()
        
        # Get unique users
        users = df['user_id'].unique()
        logger.info(f"Evaluating {len(users)} users")
        
        recalls = []
        users_with_relevant_items = 0
        
        for user_id in users:
            user_data = df[df['user_id'] == user_id].copy()
            
            # Find all relevant items for this user
            total_relevant = (user_data['actual_rating'] >= threshold).sum()
            
            # Skip users with no relevant items
            if total_relevant == 0:
                continue
            
            users_with_relevant_items += 1
            
            # Sort by predicted rating and take top-K
            top_k_items = user_data.sort_values('predicted_rating', ascending=False).head(k)
            
            # Count relevant items in top-K
            relevant_in_top_k = (top_k_items['actual_rating'] >= threshold).sum()
            
            # Calculate recall for this user
            user_recall = relevant_in_top_k / total_relevant
            recalls.append(user_recall)
        
        # Average recall across users with relevant items
        if recalls:
            mean_recall = np.mean(recalls)
        else:
            logger.warning(f"No users with relevant items (threshold={threshold})")
            mean_recall = 0.0
        
        logger.info(f"Recall@{k}: {mean_recall:.4f}")
        logger.info(f"  Users with relevant items: {users_with_relevant_items}")
        logger.info(f"  Average relevant items found: {np.mean(recalls) if recalls else 0:.2f}")
        
        return float(mean_recall)
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except TypeError as e:
        logger.error(f"Type error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calculating Recall@{k}: {e}")
        raise


if __name__ == "__main__":
    """
    Example usage and testing of the evaluator module.
    """
    try:
        print("\n" + "=" * 70)
        print("RECOMMENDER SYSTEM EVALUATION - EXAMPLE USAGE")
        print("=" * 70 + "\n")
        
        # Create sample predictions DataFrame
        print("1. Creating sample predictions data...")
        predictions = pd.DataFrame({
            'user_id': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3],
            'item_id': list(range(1, 19)),
            'actual_rating': [5, 4, 4, 3, 2, 1, 5, 5, 3, 2, 2, 1, 4, 4, 3, 3, 2, 1],
            'predicted_rating': [4.8, 4.2, 3.9, 3.1, 2.2, 1.1, 4.9, 4.7, 3.1, 2.2, 2.1, 1.2, 3.9, 3.8, 3.1, 2.9, 2.2, 1.1]
        })
        print(f"   ✓ Created {len(predictions)} predictions for {predictions['user_id'].nunique()} users\n")
        
        print("2. Calculating rating prediction metrics...")
        rmse = calculate_rmse(predictions)
        mae = calculate_mae(predictions)
        print(f"   ✓ RMSE: {rmse:.4f}")
        print(f"   ✓ MAE: {mae:.4f}\n")
        
        print("3. Calculating ranking metrics (threshold=4.0)...")
        precision_5 = precision_at_k(predictions, k=5, threshold=4.0)
        precision_10 = precision_at_k(predictions, k=10, threshold=4.0)
        recall_5 = recall_at_k(predictions, k=5, threshold=4.0)
        recall_10 = recall_at_k(predictions, k=10, threshold=4.0)
        print(f"   ✓ Precision@5: {precision_5:.4f}")
        print(f"   ✓ Precision@10: {precision_10:.4f}")
        print(f"   ✓ Recall@5: {recall_5:.4f}")
        print(f"   ✓ Recall@10: {recall_10:.4f}\n")
        
        print("4. Interpretation of results:")
        print(f"   - Average prediction error: ±{mae:.2f} stars (MAE)")
        print(f"   - Top-5 recommendations: {precision_5*100:.1f}% are good (Precision)")
        print(f"   - Top-5 coverage: Found {recall_5*100:.1f}% of all good items (Recall)\n")
        
        print("5. Predictions sample:")
        print(predictions.head(10).to_string())
        
        print("\n" + "=" * 70)
        print("✓ Evaluation completed successfully!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Failed during example usage")
