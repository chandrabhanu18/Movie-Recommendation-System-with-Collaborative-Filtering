from src.data_loader import load_ratings
from src.train_test_split import temporal_train_test_split
from src.user_based_cf import train_user_based_model, get_all_predictions

df = load_ratings("data/ml-100k")

train_df, test_df = temporal_train_test_split(df)

model = train_user_based_model(train_df)

predictions = get_all_predictions(model, test_df)

print(predictions.head())
print("Total predictions:", len(predictions))
