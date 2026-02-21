# Movie Recommendation System

A production-grade Movie Recommendation System built using Collaborative
Filtering and Matrix Factorization techniques. This project includes
User-Based CF, Item-Based CF, and SVD (Matrix Factorization) models,
along with full evaluation, Docker support, and reproducible setup.

------------------------------------------------------------------------

## Project Overview

This system recommends movies to users based on historical rating data
using:

-   User-Based Collaborative Filtering
-   Item-Based Collaborative Filtering
-   Matrix Factorization (SVD)
-   Evaluation Metrics: RMSE, MAE, Precision@10, Recall@10
-   Dockerized deployment for reproducibility

Dataset used: MovieLens 100K

------------------------------------------------------------------------

## Project Structure

    Movie Recommendation/
    │
    ├── data/
    │   └── ml-100k/
    │
    ├── src/
    │   ├── data_loader.py
    │   ├── train_test_split.py
    │   ├── user_based_cf.py
    │   ├── item_based_cf.py
    │   ├── matrix_factorization.py
    │   ├── evaluator.py
    │   └── recommender.py
    │
    ├── main.py
    ├── requirements.txt
    ├── Dockerfile
    ├── docker-compose.yml
    └── README.md

------------------------------------------------------------------------

## Models Implemented

### 1. User-Based Collaborative Filtering

-   Uses similarity between users
-   Algorithm: KNNBasic
-   RMSE: \~1.07

### 2. Item-Based Collaborative Filtering

-   Uses similarity between items
-   Algorithm: KNNBasic
-   RMSE: \~1.14

### 3. Matrix Factorization (SVD)

-   Learns latent factors
-   Best performing model
-   RMSE: \~0.98

------------------------------------------------------------------------

## Evaluation Metrics

-   RMSE (Root Mean Squared Error)
-   MAE (Mean Absolute Error)
-   Precision@10
-   Recall@10

------------------------------------------------------------------------

## Installation (Local)

### Step 1: Create Environment

    conda create -n recommender python=3.10 -y
    conda activate recommender

### Step 2: Install Dependencies

    pip install -r requirements.txt

### Step 3: Run Project

    python main.py

------------------------------------------------------------------------

## Docker Setup

### Build Docker Image

    docker build -t movie-recommender .

### Run Container

    docker run movie-recommender

### Using Docker Compose

    docker-compose up --build

------------------------------------------------------------------------

## Output Example

Top recommendations:

-   Casablanca (1942)
-   Rear Window (1954)
-   Good Will Hunting (1997)
-   Sunset Blvd. (1950)

------------------------------------------------------------------------

## Results

  Model           RMSE   MAE
  --------------- ------ ------
  User-Based CF   1.07   0.85
  Item-Based CF   1.14   0.90
  SVD             0.98   0.78

------------------------------------------------------------------------

## Key Features

-   Clean modular architecture
-   Multiple recommendation algorithms
-   Proper evaluation metrics
-   Docker support
-   Reproducible environment

------------------------------------------------------------------------

## Author

Chandra Mandiga


------------------------------------------------------------------------

## License

Educational Use Only
