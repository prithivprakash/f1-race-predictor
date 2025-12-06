import fastf1
import pandas as pd
import joblib
import numpy as np

fastf1.Cache.enable_cache("./fastf1_cache")

# ------------------------------------------
# 1) FINAL CORRECT FEATURE ORDER (TRAINED MODEL)
# ------------------------------------------
FEATURE_ORDER = [
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

# Load trained model
model = joblib.load("./models/model_fp2_sprint_xgb.joblib")


# ------------------------------------------
# 2) GET FP2 OR SPRINT PACE
# ------------------------------------------
def get_fp2_or_sprint_pace(year, race):
    """
    Returns FP2 or Sprint pace features:
    avg, median, laps, ratio
    If neither exists → fallback (None)
    """
    # 1: Try Sprint
    try:
        sprint = fastf1.get_session(year, race, "Sprint")
        sprint.load()
        laps = sprint.laps.pick_quicklaps()
        print("Using SPRINT pace")

        avg = laps.groupby("Driver")["LapTime"].mean().dt.total_seconds()
        med = laps.groupby("Driver")["LapTime"].median().dt.total_seconds()
        counts = laps.groupby("Driver")["LapTime"].count()

        return avg, med, counts

    except:
        pass

    # 2: Try FP2
    try:
        fp2 = fastf1.get_session(year, race, "FP2")
        fp2.load()
        laps = fp2.laps.pick_quicklaps()
        print("Using FP2 pace")

        avg = laps.groupby("Driver")["LapTime"].mean().dt.total_seconds()
        med = laps.groupby("Driver")["LapTime"].median().dt.total_seconds()
        counts = laps.groupby("Driver")["LapTime"].count()

        return avg, med, counts

    except:
        pass

    # 3: fallback = None
    print("No FP2/Sprint → fallback")
    return None, None, None


# ------------------------------------------
# 3) BUILD FEATURES FOR PREDICTION
# ------------------------------------------
def build_features(year, race_name):

    # Load qualifying session
    quali = fastf1.get_session(year, race_name, "Q")
    quali.load()

    q = quali.results

    # Base quali dataframe
    df = pd.DataFrame({
        "driver": q["Abbreviation"],
        "team": q["TeamName"],
        "quali_position": q["Position"],
        "quali_time_sec": q["Q3"].fillna(q["Q2"]).fillna(q["Q1"]).dt.total_seconds()
    })

    # Delta to pole
    pole = df["quali_time_sec"].min()
    df["delta_to_pole"] = df["quali_time_sec"] - pole

    # Normalized
    df["quali_normalized"] = df["quali_position"] / 20.0

    # FP2/Sprint pace
    fp2_avg, fp2_med, fp2_laps = get_fp2_or_sprint_pace(year, race_name)

    if fp2_avg is not None:
        df["pace_fp2sprint_avg"] = df["driver"].map(fp2_avg)
        df["pace_fp2sprint_med"] = df["driver"].map(fp2_med)
        df["pace_fp2sprint_laps"] = df["driver"].map(fp2_laps)
    else:
        # fallback: use quali time as placeholder
        df["pace_fp2sprint_avg"] = df["quali_time_sec"].mean()
        df["pace_fp2sprint_med"] = df["quali_time_sec"].mean()
        df["pace_fp2sprint_laps"] = 0

    # Derived FP2 ratios
    df["pace_fp2sprint_ratio"] = (
        df["pace_fp2sprint_avg"] / df["pace_fp2sprint_med"]
    ).replace([np.inf, -np.inf], 1).fillna(1)

    # Base pace from quali (fallback)
    df["avg_race_pace"] = df["pace_fp2sprint_avg"]
    df["median_race_pace"] = df["pace_fp2sprint_med"]
    df["pace_ratio"] = df["avg_race_pace"] / df["median_race_pace"]

    # Race variability = difference between FP2 and quali
    df["race_variability"] = abs(df["avg_race_pace"] - df["quali_time_sec"])

    # Strategy assumptions
    df["stint_count"] = 3
    df["pit_stops"] = 2

    # FINAL: return dataframe
    return df


# ------------------------------------------
# 4) PREDICT RACE RESULT
# ------------------------------------------
def predict_race(year, race_name):

    df = build_features(year, race_name)

    # **FORCE EXACT FEATURE ORDER**
    X = df[FEATURE_ORDER]

    # XGBoost prediction
    df["predicted_position"] = model.predict(X)
    df["predicted_position"] = df["predicted_position"].clip(1, 20)

    # Ranking + confidence
    df = df.sort_values("predicted_position").reset_index(drop=True)
    best = df["predicted_position"].iloc[0]
    df["confidence"] = 1 / (1 + (df["predicted_position"] - best))
    df["confidence"] /= df["confidence"].max()

    return df


# ------------------------------------------
# 5) Runner
# ------------------------------------------
if __name__ == "__main__":
    result = predict_race(2025, "Abu Dhabi")

    print("\n=== PREDICTED RACE ORDER ===")
    print(result[["driver", "team", "predicted_position", "confidence"]])

    print("\n--- Winner Explanation ---")
    print(result.iloc[0])
