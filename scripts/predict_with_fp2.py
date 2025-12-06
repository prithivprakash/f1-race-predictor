import fastf1
import pandas as pd
import joblib
import numpy as np

fastf1.Cache.enable_cache("./fastf1_cache")

MODEL_FILE = "./models/model_fp2_sprint_xgb.joblib"

# Feature columns (must match training)
FEATURE_COLS = [
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
    "pit_stops",
]


# --------------------------------------------------
#   LOAD QUALIFYING DATA
# --------------------------------------------------
def load_qualifying(year, race):
    q = fastf1.get_session(year, race, "Q")
    q.load()
    res = q.results

    best = res["Q3"].fillna(res["Q2"]).fillna(res["Q1"])

    df = pd.DataFrame({
        "driver": res["Abbreviation"],
        "team": res["TeamName"],
        "quali_position": res["Position"],
        "quali_time_sec": best.dt.total_seconds()
    })

    pole = df["quali_time_sec"].min()
    df["delta_to_pole"] = df["quali_time_sec"] - pole

    return df


# --------------------------------------------------
#   LOAD FP2 (IF EXISTS)
# --------------------------------------------------
def load_fp2(year, race):
    try:
        fp2 = fastf1.get_session(year, race, "FP2")
        fp2.load()

        laps = fp2.laps.pick_quicklaps()
        g = laps.groupby("Driver")["LapTime"]

        return (
            g.mean().dt.total_seconds(),
            g.median().dt.total_seconds(),
            g.count(),
        )
    except:
        return None, None, None


# --------------------------------------------------
#   LOAD SPRINT (FALLBACK IF NO FP2)
# --------------------------------------------------
def load_sprint(year, race):
    try:
        sp = fastf1.get_session(year, race, "Sprint")
        sp.load()

        laps = sp.laps.pick_quicklaps()
        g = laps.groupby("Driver")["LapTime"]

        return (
            g.mean().dt.total_seconds(),
            g.median().dt.total_seconds(),
            g.count(),
        )
    except:
        return None, None, None


# --------------------------------------------------
#   BUILD FULL FEATURE DATAFRAME
# --------------------------------------------------
def build_features(year, race):
    quali = load_qualifying(year, race)

    fp2_avg, fp2_med, fp2_laps = load_fp2(year, race)
    sp_avg, sp_med, sp_laps = load_sprint(year, race)

    # Merge pace data
    def choose(row, fp2_col, sp_col):
        if fp2_col is not None and row["driver"] in fp2_col.index:
            return fp2_col[row["driver"]]
        if sp_col is not None and row["driver"] in sp_col.index:
            return sp_col[row["driver"]]
        return row["quali_time_sec"]

    quali["pace_fp2sprint_avg"] = quali.apply(lambda r: choose(r, fp2_avg, sp_avg), axis=1)
    quali["pace_fp2sprint_med"] = quali.apply(lambda r: choose(r, fp2_med, sp_med), axis=1)
    quali["pace_fp2sprint_laps"] = quali.apply(lambda r: choose(r, fp2_laps, sp_laps), axis=1)

    # Replace missing lap counts
    quali["pace_fp2sprint_laps"].fillna(quali["pace_fp2sprint_laps"].median(), inplace=True)

    # Default placeholders (because race hasn't happened yet)
    quali["avg_race_pace"] = quali["pace_fp2sprint_avg"]
    quali["median_race_pace"] = quali["pace_fp2sprint_med"]
    quali["pace_ratio"] = quali["avg_race_pace"] / quali["median_race_pace"]
    quali["race_variability"] = 0.0  # unknown before race

    # Engineering
    quali["quali_normalized"] = quali["quali_position"] / 20
    quali["pace_fp2sprint_ratio"] = (
        quali["pace_fp2sprint_avg"] / quali["pace_fp2sprint_med"]
    )

    # Assume average stints & pit stops (better than constant values)
    quali["stint_count"] = 3
    quali["pit_stops"] = 2

    return quali


# --------------------------------------------------
#   PREDICT FUNCTION
# --------------------------------------------------
def predict_race(year, race):
    df = build_features(year, race)
    model = joblib.load(MODEL_FILE)

    X = df[FEATURE_COLS]
    df["predicted_position"] = model.predict(X).clip(1, 20)

    # Ranking
    df = df.sort_values("predicted_position").reset_index(drop=True)

    # Confidence score
    best = df["predicted_position"].iloc[0]
    df["confidence_raw"] = 1 / (1 + (df["predicted_position"] - best))
    df["confidence"] = df["confidence_raw"] / df["confidence_raw"].max()

    return df


# --------------------------------------------------
#   WINNER EXPLANATION
# --------------------------------------------------
def explain(row):
    return f"""
=== WINNER PREDICTION EXPLANATION ===

Driver: {row['driver']}
Team: {row['team']}
Predicted Finish: {row['predicted_position']:.2f}
Confidence: {row['confidence']:.2f}

Key Influences:
• Quali Position: {row['quali_position']}
• Delta to Pole: {row['delta_to_pole']:.3f}
• FP2/Sprint Avg Pace: {row['pace_fp2sprint_avg']:.3f}
• FP2/Sprint Med Pace: {row['pace_fp2sprint_med']:.3f}
• FP2/Sprint Lap Count: {row['pace_fp2sprint_laps']}
• Pace Ratio: {row['pace_ratio']:.3f}
• Expected Stints: {row['stint_count']}
• Expected Pit Stops: {row['pit_stops']}

========================================
"""


# --------------------------------------------------
#   MAIN
# --------------------------------------------------
if __name__ == "__main__":
    result = predict_race(2025, "Abu Dhabi")

    print("\n=== PREDICTED RACE ORDER ===")
    print(result[["driver", "team", "predicted_position", "confidence"]])

    print(explain(result.iloc[0]))
