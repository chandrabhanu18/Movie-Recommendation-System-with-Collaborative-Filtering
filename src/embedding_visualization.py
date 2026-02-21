"""
SVD Embedding Visualization Module
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


def visualize_item_embeddings(
    svd_model,
    movies_df,
    n_components=2,
    save_path="svd_embeddings.png"
):
    """
    Visualize item embeddings using PCA.
    """

    logger.info("Extracting item embeddings from SVD model")

    item_embeddings = svd_model.qi

    logger.info("Reducing dimensions using PCA")

    pca = PCA(n_components=n_components)

    reduced = pca.fit_transform(item_embeddings)

    embedding_df = pd.DataFrame(
        reduced,
        columns=["x", "y"]
    )

    embedding_df["item_id"] = movies_df["item_id"].values[:len(embedding_df)]
    embedding_df["title"] = movies_df["title"].values[:len(embedding_df)]

    plt.figure(figsize=(12, 8))

    plt.scatter(
        embedding_df["x"],
        embedding_df["y"],
        alpha=0.6
    )

    plt.title("SVD Item Embeddings Visualization")
    plt.xlabel("Latent Factor 1")
    plt.ylabel("Latent Factor 2")

    plt.savefig(save_path)
    plt.close()

    logger.info(f"Saved embedding visualization to {save_path}")

    return embedding_df