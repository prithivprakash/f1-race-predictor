import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

INPUT_FILE = "./data/training_preprocessed.csv"
MODEL_FILE = "./model_fp2_sprint_xgb.joblib"


def train_model():
    print("Loading dataset...")
    df = pd.read_csv(INPUT_FILE)

    # ================================
    # FEATURE COLUMNS
    # ================================
    feature_cols = [
        "quali_position",
        "quali_normalized",
        "quali_time_sec",
        "delta_to_pole",

        "avg_race_pace",
        "median_race_pace",
        "pace_ratio",
        "race_variability",

        "pace_fp2sprint_avg",
        "pace_fp2sprint_med",
        "pace_fp2sprint_laps",
        "pace_fp2sprint_ratio",

        "stint_count",
        "pit_stops"
    ]

    target_col = "final_position"

    X = df[feature_cols]
    y = df[target_col]

    # ================================
    # TRAIN / VALIDATION SPLIT
    # ================================
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, shuffle=True
    )

    # ================================
    # XGBOOST REGRESSOR
    # tuned for ranking-like predictions
    # ================================
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1.2,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # ================================
    # METRICS
    # ================================
    val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print("\n========================")
    print(f"Validation MAE  = {mae:.3f}")
    print(f"Validation RMSE = {rmse:.3f}")
    print("========================\n")

    # ================================
    # SAVE MODEL
    # ================================
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved → {MODEL_FILE}")



if __name__ == "__main__":
    train_model()
