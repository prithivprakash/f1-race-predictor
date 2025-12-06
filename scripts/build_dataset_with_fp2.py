import fastf1
import pandas as pd
from tqdm import tqdm

fastf1.Cache.enable_cache("./fastf1_cache")


# =========================
#   LOAD QUALIFYING
# =========================
def load_qualifying(year, gp):
    session = fastf1.get_session(year, gp, "Q")
    session.load()

    q = session.results
    best = q["Q3"].fillna(q["Q2"]).fillna(q["Q1"])

    df = pd.DataFrame({
        "year": year,
        "race": gp,
        "driver": q["Abbreviation"],
        "team": q["TeamName"],
        "quali_position": q["Position"],
        "quali_time_sec": best.dt.total_seconds()
    })

    pole = df["quali_time_sec"].min()
    df["delta_to_pole"] = df["quali_time_sec"] - pole

    return df


# =========================
#   LOAD FP2 (IF EXISTS)
# =========================
def load_fp2(year, gp):
    try:
        fp2 = fastf1.get_session(year, gp, "FP2")
        fp2.load()

        laps = fp2.laps.pick_quicklaps()

        df = laps.groupby("Driver")["LapTime"]
        avg = df.mean().dt.total_seconds()
        med = df.median().dt.total_seconds()
        count = df.count()

        print(f"Using FP2 for {gp} {year}")
        return avg, med, count

    except Exception:
        print(f"No FP2 → will use Sprint or fallback for {gp} {year}")
        return None, None, None


# =========================
#   LOAD SPRINT (IF EXISTS)
# =========================
def load_sprint(year, gp):
    try:
        sprint = fastf1.get_session(year, gp, "Sprint")
        sprint.load()

        laps = sprint.laps.pick_quicklaps()

        df = laps.groupby("Driver")["LapTime"]
        avg = df.mean().dt.total_seconds()
        med = df.median().dt.total_seconds()
        count = df.count()

        print(f"Using Sprint for {gp} {year}")
        return avg, med, count

    except Exception:
        print(f"No Sprint for {gp} {year}")
        return None, None, None


# =========================
#   LOAD RACE RESULTS + PACE
# =========================
def load_race(year, gp):
    session = fastf1.get_session(year, gp, "R")
    session.load()

    laps = session.laps
    results = session.results

    rows = []

    for drv_num in results["DriverNumber"].unique():
        dl = laps.pick_driver(drv_num)

        if dl.empty:
            continue

        avg = dl["LapTime"].dt.total_seconds().mean()
        med = dl["LapTime"].dt.total_seconds().median()
        stint = dl["Stint"].max()
        pitstops = dl["PitOutTime"].notna().sum()

        result_row = results[results["DriverNumber"] == drv_num].iloc[0]

        rows.append({
            "driver": result_row["Abbreviation"],
            "team": result_row["TeamName"],
            "avg_race_pace": avg,
            "median_race_pace": med,
            "stint_count": stint,
            "pit_stops": pitstops,
            "final_position": int(result_row["Position"])
        })

    return pd.DataFrame(rows)


# =========================
#   BUILD FULL DATASET
# =========================
def build_full_dataset(years):
    full = []

    for year in years:
        races = fastf1.get_event_schedule(year)
        gp_list = races["EventName"].tolist()

        print(f"\n=== Building {year} dataset ===\n")

        for gp in tqdm(gp_list):
            try:
                quali = load_qualifying(year, gp)

                fp2_avg, fp2_med, fp2_cnt = load_fp2(year, gp)
                sprint_avg, sprint_med, sprint_cnt = load_sprint(year, gp)

                race = load_race(year, gp)

                df = quali.merge(race, on=["driver", "team"], how="inner")

                # map FP2
                df["fp2_avg_pace"] = df["driver"].map(fp2_avg) if fp2_avg is not None else None
                df["fp2_med_pace"] = df["driver"].map(fp2_med) if fp2_med is not None else None
                df["fp2_laps"] = df["driver"].map(fp2_cnt) if fp2_cnt is not None else None

                # map Sprint
                df["sprint_avg_pace"] = df["driver"].map(sprint_avg) if sprint_avg is not None else None
                df["sprint_med_pace"] = df["driver"].map(sprint_med) if sprint_med is not None else None
                df["sprint_laps"] = df["driver"].map(sprint_cnt) if sprint_cnt is not None else None

                full.append(df)

            except Exception as e:
                print(f"Skipping {gp} {year}: {e}")

    final = pd.concat(full, ignore_index=True)
    final.to_csv("./data/training_fp2_sprint_2024_2025.csv", index=False)

    print("\n==========================")
    print(" Saved → training_fp2_sprint_2024_2025.csv")
    print("==========================\n")

    return final


if __name__ == "__main__":
    build_full_dataset([2024, 2025])
