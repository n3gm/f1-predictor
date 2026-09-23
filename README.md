# 🏎️ Formula 1 Race Winner Predictor

An end-to-end Machine Learning pipeline that predicts the winner of Formula 1 Grand Prix races using qualifying telemetry, grid positions, and rolling momentum features.

---

## 📌 Project Overview
Predicting the winner of an F1 race cannot rely on static driver names because car performance and team momentum shift across seasons. This project builds a production-ready ML workflow that transforms raw timing and telemetry data into rolling performance signals, providing race predictions before lights out on Sunday.

---

## 🏗️ Architecture & Pipeline

```text
[FastF1 API] 
     │ (Raw Sessions: Qualifying & Race)
     ▼
[Data Pipeline & Cleaning]
     │ (Missing values, time delta conversion)
     ▼
[Feature Engineering]
     │ (Shifted rolling averages, delta-to-pole)
     ▼
[Model Training & Evaluation]
     │ (Random Forest / GBDT with Time-based Split)
     ▼
[Streamlit Dashboard]
       (Interactive grid simulator & probability ranker)

```
---


#  Key Engineered Features
GridPosition: Official starting position on the grid.

DeltaToPole: Gap in seconds to the fastest qualifying lap (measures raw single-lap pace).

Driver_Avg_Finish_Last3: Shifted 3-race rolling average of driver finishes (captures driver form without data leakage).

Team_Avg_Finish_Last3: Shifted 3-race rolling average of constructor finishes (measures car development and team momentum).

---

#  Evaluation Strategy
Because each race has exactly 1 winner and 19 losers (heavy class imbalance), standard 0.5 classification thresholding fails.

The model is evaluated using:

Per-Race Top-1 Accuracy: Did the driver with the highest predicted win probability actually win the race?

Temporal Split: The model is trained on historical rounds and tested exclusively on future unseen rounds to simulate real deployment.

---

# Tech Stack
- Language: Python
- Data Sources: FastF1, Ergast API
- Data Manipulation: Pandas, NumPy
- Machine Learning: Scikit-Learn
- Dashboard & Serving: Streamlit
