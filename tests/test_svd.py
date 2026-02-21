from src.data_loader import load_ratings
from src.train_test_split import temporal_train_test_split
from src.matrix_factorization import train_svd_model, get_all_predictions

df = load_ratings("data/ml-100k")

train_df, test_df = temporal_train_test_split(df)

model = train_svd_model(train_df)

predictions = get_all_predictions(model, test_df)

print(predictions.head())
print("Total:", len(predictions))
