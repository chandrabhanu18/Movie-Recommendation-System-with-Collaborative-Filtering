"""
Temporal Train-Test Split Module for Recommender Systems

This module provides functionality for splitting rating data into training and testing
sets using a temporal (time-aware) approach. This is critical for recommender systems
because:

1. **Realistic Evaluation**: Simulates real-world scenarios where we train on historical
   data and predict future ratings. This prevents data leakage from future information.

2. **Temporal Dynamics**: Users' preferences evolve over time. Evaluating on future ratings
   tests the model's ability to adapt to changing preferences.

3. **Avoids Bias**: Time-aware splitting ensures we don't accidentally use future information
   to predict the past, which would artificially inflate model performance metrics.

4. **User Coverage**: Ensures every user appears in the training set, enabling personalized
   recommendations. Users with insufficient data are excluded to maintain reliability.

5. **Chronological Validity**: Maintains the temporal order of interactions, crucial for
   understanding user behavior patterns and recommendation sequences.

Example Usage:
    >>> import pandas as pd
    >>> from pathlib import Path
    >>> import data_loader
    >>> import train_test_split
    >>> 
    >>> # Load ratings data
    >>> data_path = Path("data/ml-100k")
    >>> ratings_df = data_loader.load_ratings(str(data_path))
    >>> 
    >>> # Perform temporal train-test split
    >>> train_df, test_df = train_test_split.temporal_train_test_split(
    ...     ratings_df,
    ...     test_size=0.2
    ... )
    >>> print(f"Training set: {len(train_df)} ratings")
    >>> print(f"Test set: {len(test_df)} ratings")
"""

import logging
from typing import Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MIN_USER_RATINGS = 5  # Minimum number of ratings per user


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split rating data into training and testing sets using temporal ordering.
    
    This function performs a time-aware train-test split that respects the chronological
    order of user ratings. For each user, earlier ratings go to the training set and
    more recent ratings go to the testing set. This approach is essential for recommender
    system evaluation as it:
    
    - Prevents information leakage from future ratings
    - Maintains realistic evaluation conditions
    - Ensures users' preference evolution is respected
    - Simulates how the model would be used in production
    
    Args:
        df (pd.DataFrame): Input DataFrame with columns:
                          'user_id', 'item_id', 'rating', 'timestamp', 'title'
                          Must be sorted by timestamp (not required, but recommended).
        
        test_size (float): Fraction of each user's most recent ratings to use for testing.
                          Must be between 0.0 and 1.0. Default is 0.2 (20%).
                          Example: test_size=0.2 means the last 20% of each user's ratings
                                  go to the test set.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - train_df: DataFrame with earlier ratings (80% for test_size=0.2)
            - test_df: DataFrame with more recent ratings (20% for test_size=0.2)
            
            Both DataFrames maintain the original column structure and order.
    
    Raises:
        ValueError: If df is empty, test_size is invalid, or required columns are missing.
        TypeError: If inputs have incorrect types.
        Exception: For other unexpected errors during splitting.
    
    Notes:
        - Users with fewer than MIN_USER_RATINGS (5) ratings are excluded from both sets
        - This ensures training data quality and meaningful test sets
        - Users are not filtered out entirely; instead, ratings are split by user
        - The resulting test_df may contain some users that don't appear in train_df
          if they have few ratings (the split might place all ratings in train or test)
    
    Example:
        >>> ratings_df = load_ratings("data/ml-100k")
        >>> train_df, test_df = temporal_train_test_split(ratings_df, test_size=0.2)
        >>> 
        >>> # Verify the split
        >>> print(f"Training ratings: {len(train_df)}")
        >>> print(f"Test ratings: {len(test_df)}")
        >>> 
        >>> # Check user coverage
        >>> train_users = set(train_df['user_id'].unique())
        >>> test_users = set(test_df['user_id'].unique())
        >>> print(f"Users in train set: {len(train_users)}")
        >>> print(f"Users in test set: {len(test_users)}")
        >>> 
        >>> # Sample user's temporal split
        >>> sample_user = train_df['user_id'].iloc[0]
        >>> user_train = train_df[train_df['user_id'] == sample_user].sort_values('timestamp')
        >>> user_test = test_df[test_df['user_id'] == sample_user].sort_values('timestamp')
        >>> print(f"User {sample_user} - Train: {len(user_train)}, Test: {len(user_test)}")
    """
    try:
        # Validate input DataFrame
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        # Validate test_size parameter
        if not isinstance(test_size, (int, float)):
            raise TypeError(f"test_size must be numeric, got {type(test_size)}")
        
        if not (0.0 < test_size < 1.0):
            raise ValueError(
                f"test_size must be between 0.0 and 1.0 (exclusive), got {test_size}"
            )
        
        # Validate required columns
        required_columns = ['user_id', 'item_id', 'rating', 'timestamp', 'title']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(
                f"Input DataFrame missing required columns: {missing_columns}. "
                f"Has columns: {list(df.columns)}"
            )
        
        logger.info(f"Starting temporal train-test split with test_size={test_size}")
        logger.info(f"Total input ratings: {len(df)}")
        
        # Create a working copy to avoid modifying original
        df_work = df.copy()
        
        # Ensure data is sorted by timestamp for consistent temporal ordering
        df_work = df_work.sort_values('timestamp').reset_index(drop=True)
        logger.info("Data sorted by timestamp")
        
        # Group by user and filter out users with insufficient ratings
        user_rating_counts = df_work.groupby('user_id').size()
        valid_users = user_rating_counts[user_rating_counts >= MIN_USER_RATINGS].index.tolist()
        
        logger.info(f"Total unique users: {len(user_rating_counts)}")
        logger.info(f"Users with >= {MIN_USER_RATINGS} ratings: {len(valid_users)}")
        logger.info(
            f"Excluded users with < {MIN_USER_RATINGS} ratings: "
            f"{len(user_rating_counts) - len(valid_users)}"
        )
        
        # Filter to only valid users
        df_filtered = df_work[df_work['user_id'].isin(valid_users)].copy()
        logger.info(f"Total ratings after filtering: {len(df_filtered)}")
        
        # Split data for each user
        train_list = []
        test_list = []
        
        for user_id in valid_users:
            user_data = df_filtered[df_filtered['user_id'] == user_id].sort_values('timestamp')
            num_ratings = len(user_data)
            
            # Calculate split index
            # For example: 10 ratings with test_size=0.2 -> split at index 8
            split_idx = int(np.ceil(num_ratings * (1 - test_size)))
            
            # Ensure at least one rating in training set
            split_idx = max(1, split_idx)
            # Ensure at least one rating in test set (if possible)
            split_idx = min(split_idx, num_ratings - 1)
            
            # Split user's ratings
            user_train = user_data.iloc[:split_idx]
            user_test = user_data.iloc[split_idx:]
            
            train_list.append(user_train)
            test_list.append(user_test)
        
        # Concatenate all user splits
        train_df = pd.concat(train_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)
        
        # Reset indices
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        # Verify result structure
        assert set(train_df.columns) == set(df.columns), "Train DataFrame column mismatch"
        assert set(test_df.columns) == set(df.columns), "Test DataFrame column mismatch"
        
        # Log summary statistics
        logger.info("=" * 70)
        logger.info("TEMPORAL TRAIN-TEST SPLIT SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total ratings (original):    {len(df):,}")
        logger.info(f"Total ratings (filtered):    {len(df_filtered):,}")
        logger.info(f"Training set size:           {len(train_df):,} ratings")
        logger.info(f"Test set size:               {len(test_df):,} ratings")
        logger.info(f"Train/Test ratio:            {len(train_df)/len(test_df):.2f}:1")
        logger.info(f"Users in training set:       {train_df['user_id'].nunique()}")
        logger.info(f"Unique users overall:        {len(valid_users)}")
        
        # Additional statistics
        avg_train_per_user = len(train_df) / len(valid_users)
        avg_test_per_user = len(test_df) / len(valid_users)
        logger.info(f"Avg train ratings per user:  {avg_train_per_user:.2f}")
        logger.info(f"Avg test ratings per user:   {avg_test_per_user:.2f}")
        logger.info("=" * 70)
        
        return train_df, test_df
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except TypeError as e:
        logger.error(f"Type error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during temporal train-test split: {e}")
        raise


if __name__ == "__main__":
    """
    Example usage and testing of the temporal train-test split module.
    """
    try:
        from pathlib import Path
        import sys
        
        # Add parent directory to path to import data_loader
        sys.path.insert(0, str(Path(__file__).parent))
        import data_loader
        
        print("\n" + "=" * 70)
        print("TEMPORAL TRAIN-TEST SPLIT - EXAMPLE USAGE")
        print("=" * 70 + "\n")
        
        # Load data
        data_path = Path(__file__).parent.parent / "data" / "ml-100k"
        
        if not data_path.exists():
            print(f"❌ Dataset not found at {data_path}")
            print("Please ensure the MovieLens 100K dataset is located at:")
            print("  data/ml-100k/")
            sys.exit(1)
        
        print("1. Loading ratings data...")
        ratings_df = data_loader.load_ratings(str(data_path))
        print(f"   ✓ Loaded {len(ratings_df)} ratings\n")
        
        # Perform temporal split
        print("2. Performing temporal train-test split (test_size=0.2)...")
        train_df, test_df = temporal_train_test_split(ratings_df, test_size=0.2)
        print(f"   ✓ Split completed\n")
        
        # Display results
        print("3. Split Results:")
        print(f"   Training set:  {len(train_df):,} ratings")
        print(f"   Test set:      {len(test_df):,} ratings")
        print(f"   Total:         {len(train_df) + len(test_df):,} ratings\n")
        
        # User coverage analysis
        train_users = set(train_df['user_id'].unique())
        test_users = set(test_df['user_id'].unique())
        both_users = train_users & test_users
        
        print("4. User Coverage:")
        print(f"   Users in training set:     {len(train_users)}")
        print(f"   Users in test set:         {len(test_users)}")
        print(f"   Users in both sets:        {len(both_users)}")
        print(f"   Coverage percentage:       {len(train_users) / len(both_users) * 100:.1f}%\n")
        
        # Sample user analysis
        print("5. Sample User Temporal Analysis:")
        sample_user = train_df['user_id'].iloc[0]
        user_train = train_df[train_df['user_id'] == sample_user].sort_values('timestamp')
        user_test = test_df[test_df['user_id'] == sample_user].sort_values('timestamp')
        
        print(f"   User ID: {sample_user}")
        print(f"   Training ratings: {len(user_train)}")
        print(f"   Test ratings: {len(user_test)}")
        
        if len(user_train) > 0 and len(user_test) > 0:
            earliest_train = user_train['timestamp'].min()
            latest_train = user_train['timestamp'].max()
            earliest_test = user_test['timestamp'].min()
            latest_test = user_test['timestamp'].max()
            
            print(f"   Train time range: {earliest_train} to {latest_train}")
            print(f"   Test time range:  {earliest_test} to {latest_test}")
        
        print("\n" + "=" * 70)
        print("✓ Temporal train-test split completed successfully!")
        print("=" * 70 + "\n")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure data_loader.py is in the same directory")
        logger.exception("Failed during example usage")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Failed to run example usage")
