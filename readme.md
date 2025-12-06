```markdown
# 🏎️ F1 Race Predictor — FP2 + Sprint Enhanced Model (2024–2025)

An end-to-end machine learning pipeline that predicts Formula 1 race results using:
- Qualifying performance  
- FP2 race-simulation pace (or Sprint pace when FP2 is missing)  
- Custom engineered pace features  
- A trained XGBoost regression model  

This project generates **race result predictions in the terminal** and runs fully on open-source timing data via **FastF1**.

---

## 🚀 Features

### 📌 1. Full Automated Dataset Builder  
Builds a 2024 + 2025 training dataset including:
- Qualifying times  
- FP2 long-run average & median pace  
- Sprint pace as fallback  
- Race-pace features  
- Pit stop counts & stint counts  
- Delta-to-pole normalization  

### 📌 2. Data Preprocessing  
Cleans missing values, clips outliers, and performs light scaling.  
The model is **agnostic to driver/team names** — so new drivers in 2025 are handled automatically.

### 📌 3. FP2/Sprint-Aware XGBoost Model  
Trained using:
- qualy_position  
- pace_fp2sprint_avg  
- pace_fp2sprint_med  
- pace_ratio  
- race_variability  
- stint_count  
- pit_stops  
… and other engineered performance metrics.

Typical validation:  
```

MAE ≈ 2.4 – 2.6
RMSE ≈ 3.5 – 3.8

```
(Strong performance for motorsport prediction noise levels.)

### 📌 4. Terminal-based Predictor (Final Output)  
Run one script and instantly get:

- Sorted predicted race order  
- Confidence score per driver  
- Explanation of top driver  
- Full FP2/Sprint pace diagnostics  

## 🛠 Installation

### 1. Create virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

FastF1 cache will generate automatically on first run.

---

## 📊 Building the Full Dataset (2024 + 2025 FP2/Sprint)

Run:

```bash
python scripts/build_dataset_with_fp2.py
```

This produces:

```
data/training_fp2_sprint_2024_2025.csv
```

---

## 🧹 Preprocess the Data

```bash
python scripts/data_preprocessing.py
```

Output:

```
data/training_preprocessed.csv
```

---

## 🤖 Train the XGBoost Model

```bash
python scripts/train_fp2_sprint_model.py
```

Output:

```
models/model_fp2_sprint_xgb.joblib
```

---

## 🔮 Predict a Race (Final Tool)

Predict directly from qualifying + FP2/Sprint:

```bash
python scripts/predict_with_fp2.py
```

Example Output:

```
=== PREDICTED RACE ORDER ===
driver   team              predicted_position   confidence
VER      Red Bull Racing         1.63             1.00
NOR      McLaren                2.51             0.53
LEC      Ferrari                3.43             0.35
...

=== WINNER EXPLANATION ===
Driver: VER
Team: Red Bull Racing
FP2/Sprint Avg: 84.299
Delta to Pole: 0.000
Expected Pit Stops: 2
```

---

## 📌 Notes

* The predictor **does not scrape or use race results** → it predicts only from *future-facing* data (qualifying + FP2/Sprint).
* FP2 data is automatically replaced with Sprint data for sprint weekends.
* New drivers, teams, or missing sessions are handled automatically.

---

## 📮 Future Improvements

* Add Monte Carlo race simulation backend
* Strategy modeling (soft/medium/hard stint degradation)
* Weather-adjusted predictions
* A clean UI or web dashboard (Streamlit/FastAPI)

---

## 🤝 Contributing

Open to pull requests, feedback, and discussion.

---

## ⭐ If you found this project helpful

Please star ⭐ the repo — it motivates development!

---

# 🏁 Enjoy the racing & enjoy the predictions!
