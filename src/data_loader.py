"""
Data Loader Module for MovieLens 100K Dataset

This module provides functions for loading and preprocessing the MovieLens 100K dataset,
including rating data and movie information, creation of user-item matrices, and 
sparsity calculations.

Example Usage:
    >>> from pathlib import Path
    >>> import data_loader
    >>> 
    >>> # Load ratings data
    >>> data_path = Path("data/ml-100k")
    >>> ratings_df = data_loader.load_ratings(str(data_path))
    >>> print(f"Loaded {len(ratings_df)} ratings")
    >>> 
    >>> # Create user-item matrix
    >>> user_item_matrix = data_loader.create_user_item_matrix(ratings_df)
    >>> print(f"Matrix shape: {user_item_matrix.shape}")
    >>> 
    >>> # Calculate sparsity
    >>> sparsity = data_loader.calculate_sparsity(user_item_matrix)
    >>> print(f"Dataset sparsity: {sparsity:.2f}%")
"""

import logging
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ratings(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess the MovieLens 100K ratings dataset with movie titles.
    
    This function reads the u.data and u.item files from the MovieLens 100K dataset,
    merges them to include movie titles with ratings, and returns a cleaned DataFrame
    sorted by timestamp.
    
    Args:
        data_path (str): Path to the directory containing the MovieLens 100K dataset
                        (should contain u.data and u.item files).
    
    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - user_id (int): User identifier
            - item_id (int): Movie/item identifier
            - rating (int): Rating value (1-5)
            - timestamp (datetime): Timestamp of the rating
            - title (str): Movie title
    
    Raises:
        FileNotFoundError: If the data files are not found at the specified path.
        ValueError: If the data files are empty or malformed.
        Exception: For other unexpected errors during data loading.
    
    Example:
        >>> ratings_df = load_ratings("data/ml-100k")
        >>> print(ratings_df.head())
        >>> print(f"Total ratings: {len(ratings_df)}")
        >>> print(f"Date range: {ratings_df['timestamp'].min()} to {ratings_df['timestamp'].max()}")
    """
    try:
        data_dir = Path(data_path)
        ratings_file = data_dir / "u.data"
        items_file = data_dir / "u.item"
        
        # Validate file existence
        if not ratings_file.exists():
            raise FileNotFoundError(f"Ratings file not found: {ratings_file}")
        if not items_file.exists():
            raise FileNotFoundError(f"Items file not found: {items_file}")
        
        logger.info(f"Loading ratings from {ratings_file}")
        
        # Load ratings data
        # Format: user_id | item_id | rating | timestamp (tab-separated)
        ratings_df = pd.read_csv(
            ratings_file,
            sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'],
            engine='python'
        )
        
        if ratings_df.empty:
            raise ValueError("Ratings file is empty")
        
        logger.info(f"Loaded {len(ratings_df)} ratings")
        
        # Convert timestamp to datetime
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
        logger.info("Converted timestamps to datetime format")
        
        # Load movie items data
        logger.info(f"Loading movie items from {items_file}")
        
        # u.item format: item_id | title | release_date | video_release_date | imdb_url | genres...
        # We only need item_id and title
        items_df = pd.read_csv(
            items_file,
            sep='|',
            encoding='latin-1',
            usecols=[0, 1],
            names=['item_id', 'title'],
            header=None,
            engine='python'
        )
        
        if items_df.empty:
            raise ValueError("Items file is empty")
        
        logger.info(f"Loaded {len(items_df)} movie titles")
        
        # Merge ratings with movie titles
        merged_df = ratings_df.merge(items_df, on='item_id', how='left')
        logger.info("Merged ratings with movie titles")
        
        # Check for any missing titles
        missing_titles = merged_df['title'].isna().sum()
        if missing_titles > 0:
            logger.warning(f"{missing_titles} ratings have missing movie titles")
        
        # Sort by timestamp
        merged_df = merged_df.sort_values('timestamp').reset_index(drop=True)
        logger.info("Sorted data by timestamp and reset index")
        
        # Reorder columns for better readability
        merged_df = merged_df[['user_id', 'item_id', 'rating', 'timestamp', 'title']]
        
        logger.info(f"Successfully loaded and preprocessed {len(merged_df)} ratings")
        return merged_df
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading ratings: {e}")
        raise


def create_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a user-item rating matrix from a ratings DataFrame.
    
    This function pivots the ratings data to create a matrix where rows represent
    users, columns represent items (movies), and values are ratings. Missing values
    (movies not rated by users) remain as NaN.
    
    Args:
        df (pd.DataFrame): DataFrame containing at least the columns:
                          'user_id', 'item_id', and 'rating'.
    
    Returns:
        pd.DataFrame: A pivot table with:
            - Index: user_id (rows)
            - Columns: item_id (columns)
            - Values: rating (cell values)
            - NaN for user-item pairs without ratings
    
    Raises:
        KeyError: If required columns are missing from the input DataFrame.
        ValueError: If the input DataFrame is empty.
        Exception: For other unexpected errors during matrix creation.
    
    Example:
        >>> ratings_df = load_ratings("data/ml-100k")
        >>> matrix = create_user_item_matrix(ratings_df)
        >>> print(f"Matrix shape: {matrix.shape}")
        >>> print(f"Number of ratings: {matrix.notna().sum().sum()}")
        >>> # Access rating for user 1, item 10
        >>> rating = matrix.loc[1, 10] if (1 in matrix.index and 10 in matrix.columns) else None
    """
    try:
        # Validate input
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        required_columns = ['user_id', 'item_id', 'rating']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")
        
        logger.info("Creating user-item matrix")
        
        # Create pivot table
        user_item_matrix = df.pivot_table(
            index='user_id',
            columns='item_id',
            values='rating',
            aggfunc='mean'  # Use mean in case of duplicate ratings
        )
        
        logger.info(f"Created user-item matrix with shape: {user_item_matrix.shape}")
        logger.info(f"Number of users: {user_item_matrix.shape[0]}")
        logger.info(f"Number of items: {user_item_matrix.shape[1]}")
        logger.info(f"Number of ratings: {user_item_matrix.notna().sum().sum()}")
        
        return user_item_matrix
        
    except KeyError as e:
        logger.error(f"Key error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating user-item matrix: {e}")
        raise


def calculate_sparsity(matrix: pd.DataFrame) -> float:
    """
    Calculate the sparsity percentage of a user-item rating matrix.
    
    Sparsity represents the proportion of missing ratings in the matrix.
    A higher sparsity indicates more missing values, which is common in
    recommendation systems where users rate only a small fraction of available items.
    
    Formula:
        sparsity = 1 - (number of non-zero ratings / total possible ratings)
        sparsity_percentage = sparsity * 100
    
    Args:
        matrix (pd.DataFrame): User-item rating matrix with users as rows and
                              items as columns. NaN values represent missing ratings.
    
    Returns:
        float: Sparsity percentage (0-100), where:
              - 0% means all possible ratings exist (dense matrix)
              - 100% means no ratings exist (completely sparse)
    
    Raises:
        ValueError: If the input matrix is empty or has invalid dimensions.
        Exception: For other unexpected errors during calculation.
    
    Example:
        >>> ratings_df = load_ratings("data/ml-100k")
        >>> matrix = create_user_item_matrix(ratings_df)
        >>> sparsity = calculate_sparsity(matrix)
        >>> print(f"Dataset sparsity: {sparsity:.2f}%")
        >>> print(f"Density: {100 - sparsity:.2f}%")
    """
    try:
        # Validate input
        if matrix.empty:
            raise ValueError("Input matrix is empty")
        
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("Matrix has invalid dimensions")
        
        # Calculate total possible ratings
        total_elements = matrix.shape[0] * matrix.shape[1]
        
        # Calculate number of actual ratings (non-NaN values)
        num_ratings = matrix.notna().sum().sum()
        
        # Calculate sparsity
        sparsity = 1 - (num_ratings / total_elements)
        sparsity_percentage = sparsity * 100
        
        logger.info(f"Matrix dimensions: {matrix.shape[0]} users × {matrix.shape[1]} items")
        logger.info(f"Total possible ratings: {total_elements}")
        logger.info(f"Actual ratings: {num_ratings}")
        logger.info(f"Sparsity: {sparsity_percentage:.2f}%")
        
        return sparsity_percentage
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error calculating sparsity: {e}")
        raise


if __name__ == "__main__":
    """
    Example usage and testing of the data loader module.
    """
    # Example: Load and analyze MovieLens 100K dataset
    try:
        data_path = Path("data/ml-100k")
        
        print("=" * 60)
        print("MovieLens 100K Dataset Loader - Example Usage")
        print("=" * 60)
        
        # Load ratings
        print("\n1. Loading ratings data...")
        ratings_df = load_ratings(str(data_path))
        print(f"   ✓ Loaded {len(ratings_df)} ratings")
        print(f"   ✓ Columns: {list(ratings_df.columns)}")
        print(f"\n   First few ratings:")
        print(ratings_df.head())
        
        # Create user-item matrix
        print("\n2. Creating user-item matrix...")
        user_item_matrix = create_user_item_matrix(ratings_df)
        print(f"   ✓ Matrix shape: {user_item_matrix.shape}")
        
        # Calculate sparsity
        print("\n3. Calculating sparsity...")
        sparsity = calculate_sparsity(user_item_matrix)
        print(f"   ✓ Sparsity: {sparsity:.2f}%")
        print(f"   ✓ Density: {100 - sparsity:.2f}%")
        
        print("\n" + "=" * 60)
        print("Data loading completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Failed to run example usage")

