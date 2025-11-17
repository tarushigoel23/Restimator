import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv("sleep_dataset.csv")

X = data[["bedtime", "screen_time", "stress", "weather", "alarms"]]
y = data["sleep_duration"]

preprocessor = ColumnTransformer(
    transformers=[
        ("weather", OneHotEncoder(sparse_output=False, drop='if_binary'), ["weather"])
    ],
    remainder='passthrough'
)

model = Pipeline([
    ("preprocess", preprocessor),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=98
)

model.fit(X_train, y_train)

joblib.dump(model, "sleep_model.pkl")

print("Model trained and saved!")