import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "f1_artifacts.pkl")

st.set_page_config(page_title="F1 Winner Predictor", layout="wide")
st.title("🏎️ لوحة محاكاة وتوقع الفائز في سباق الفورمولا 1")

if not os.path.exists(MODEL_PATH):
    st.error("لم يتم العثور على المودل. شغّل train.py أولاً لتوليد الملف.")
    st.stop()

artifacts = joblib.load(MODEL_PATH)
model = artifacts["model"]
driver_stats = artifacts["driver_stats"]
team_stats = artifacts["team_stats"]

sample_data = [
    {
        "Driver": "Max Verstappen",
        "DriverId": "verstappen",
        "Team": "Red Bull Racing",
        "Grid": 1,
        "Delta": 0.000,
    },
    {
        "Driver": "Lando Norris",
        "DriverId": "norris",
        "Team": "McLaren",
        "Grid": 2,
        "Delta": 0.095,
    },
    {
        "Driver": "Charles Leclerc",
        "DriverId": "leclerc",
        "Team": "Ferrari",
        "Grid": 3,
        "Delta": 0.150,
    },
    {
        "Driver": "Oscar Piastri",
        "DriverId": "piastri",
        "Team": "McLaren",
        "Grid": 4,
        "Delta": 0.220,
    },
    {
        "Driver": "George Russell",
        "DriverId": "russell",
        "Team": "Mercedes",
        "Grid": 5,
        "Delta": 0.310,
    },
    {
        "Driver": "Lewis Hamilton",
        "DriverId": "hamilton",
        "Team": "Ferrari",
        "Grid": 6,
        "Delta": 0.380,
    },
]

st.subheader("تعديل مراكز الانطلاق وفوارق التوقيت للسباق القادم:")
edited_df = st.data_editor(
    pd.DataFrame(sample_data), use_container_width=True, num_rows="dynamic"
)

if st.button("توقع الفائز 🏁"):
    rows = []
    for _, r in edited_df.iterrows():
        rows.append(
            {
                "GridPosition": r["Grid"],
                "Q_Position": r["Grid"],
                "DeltaToPole": r["Delta"],
                "Driver_Avg_Finish_Last3": driver_stats.get(
                    r["DriverId"], r["Grid"]
                ),
                "Team_Avg_Finish_Last3": team_stats.get(r["Team"], r["Grid"]),
            }
        )

    preds = model.predict_proba(pd.DataFrame(rows))[:, 1]
    norm_preds = (preds / preds.sum()) * 100

    edited_df["Win Probability (%)"] = np.round(norm_preds, 1)
    results = edited_df.sort_values(
        by="Win Probability (%)", ascending=False
    ).reset_index(drop=True)

    winner = results.iloc[0]
    st.success(
        f"🏆 الفائز الأقرب: **{winner['Driver']}** بنسبة فوز **{winner['Win Probability (%)']}%**"
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(results[["Driver", "Team", "Grid", "Win Probability (%)"]])
    with c2:
        st.bar_chart(results.set_index("Driver")["Win Probability (%)"])