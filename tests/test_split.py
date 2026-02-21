from src.data_loader import load_ratings
from src.train_test_split import temporal_train_test_split

df = load_ratings("data/ml-100k")

train_df, test_df = temporal_train_test_split(df, test_size=0.2)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

print("Train latest timestamp:", train_df["timestamp"].max())
print("Test earliest timestamp:", test_df["timestamp"].min())
