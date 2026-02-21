from src.data_loader import load_ratings, create_user_item_matrix, calculate_sparsity


def main():
    print("Loading ratings...")

    df = load_ratings("data/ml-100k")

    print("\nRatings DataFrame shape:", df.shape)
    print(df.head())

    print("\nCreating user-item matrix...")

    matrix = create_user_item_matrix(df)

    print("\nUser-item matrix shape:", matrix.shape)

    sparsity = calculate_sparsity(matrix)

    print(f"\nDataset sparsity: {sparsity:.4f}")


if __name__ == "__main__":
    main()
