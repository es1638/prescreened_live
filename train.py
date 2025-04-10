
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Load historical features
df = pd.read_csv("historical_features.csv")

# Features and target
X = df.drop(columns=["target"])
y = df["target"]

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# LightGBM parameters
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "verbose": -1,
}

# Train the model
model = lgb.train(
    params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=1000,
    early_stopping_rounds=50,
)

# Save Booster model
model.save_model("lightgbm_model_converted.txt")
