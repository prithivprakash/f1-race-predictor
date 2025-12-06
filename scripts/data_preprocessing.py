import pandas as pd
import numpy as np

INPUT_FILE = "./data/training_fp2_sprint_2024_2025.csv"
OUTPUT_FILE = "./data/training_preprocessed.csv"


def preprocess():
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded dataset: {df.shape[0]} rows")

    # -------------------------------
    # 1. Fix DNFs → final_position = 20
    # -------------------------------
    df["final_position"] = df["final_position"].fillna(20).clip(1, 20)

    # -------------------------------
    # 2. Fill missing FP2/Sprint data
    # Priority: FP2 → Sprint → fallback
    # -------------------------------

    # Fallback pace = driver’s qualifying time
    fallback_pace = df["quali_time_sec"]

    def choose_value(row, fp2_col, sp_col):
        if not pd.isna(row[fp2_col]):
            return row[fp2_col]
        if not pd.isna(row[sp_col]):
            return row[sp_col]
        return row["quali_time_sec"]

    df["pace_fp2sprint_avg"] = df.apply(
        lambda r: choose_value(r, "fp2_avg_pace", "sprint_avg_pace"), axis=1
    )
    df["pace_fp2sprint_med"] = df.apply(
        lambda r: choose_value(r, "fp2_med_pace", "sprint_med_pace"), axis=1
    )
    df["pace_fp2sprint_laps"] = df.apply(
        lambda r: choose_value(r, "fp2_laps", "sprint_laps"), axis=1
    )

    # If laps missing → replace with median laps
    df["pace_fp2sprint_laps"].fillna(df["pace_fp2sprint_laps"].median(), inplace=True)

    # -------------------------------
    # 3. Replace missing race pace values
    # -------------------------------
    df["avg_race_pace"].fillna(df["avg_race_pace"].median(), inplace=True)
    df["median_race_pace"].fillna(df["median_race_pace"].median(), inplace=True)

    df["stint_count"].fillna(0, inplace=True)
    df["pit_stops"].fillna(0, inplace=True)

    # -------------------------------
    # 4. Feature engineering
    # -------------------------------

    df["quali_normalized"] = df["quali_position"] / 20

    df["pace_ratio"] = df["avg_race_pace"] / df["median_race_pace"]
    df["pace_fp2sprint_ratio"] = df["pace_fp2sprint_avg"] / df["pace_fp2sprint_med"]

    df["race_variability"] = (
        abs(df["avg_race_pace"] - df["median_race_pace"]) /
        df["median_race_pace"]
    )

    # -------------------------------
    # 5. Remove impossible values
    # -------------------------------
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    # -------------------------------
    # 6. Save preprocessed dataset
    # -------------------------------
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n==== Preprocessing Complete ====")
    print(f"Saved → {OUTPUT_FILE}")
    print(f"Final rows: {df.shape[0]}")
    print("===============================\n")


if __name__ == "__main__":
    preprocess()
