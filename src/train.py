import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "f1_processed_data.csv"
)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "f1_artifacts.pkl")


def train_and_evaluate():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    features = [
        "GridPosition",
        "Q_Position",
        "DeltaToPole",
        "Driver_Avg_Finish_Last3",
        "Team_Avg_Finish_Last3",
    ]
    target = "IsWinner"

    # تقسيم البيانات زمنياً: التدريب على ما قبل 2025/2026 والاختبار على أحدث الجولات
    split_condition = (df["Year"] == 2026) | (
        (df["Year"] == 2025) & (df["RoundNumber"] > 16)
    )

    train_df = df[~split_condition]
    test_df = df[split_condition].copy()

    X_train, y_train = train_df[features], train_df[target]
    X_test, y_test = test_df[features], test_df[target]

    print(f"بيانات التدريب: {len(X_train)} صفاً | بيانات الاختبار: {len(X_test)} صفاً")

    model = RandomForestClassifier(
        n_estimators=150, max_depth=6, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    # تقييم Per-Race Top-1
    test_df["Prob"] = model.predict_proba(X_test)[:, 1]
    correct, total = 0, 0

    for (yr, rnd), group in test_df.groupby(["Year", "RoundNumber"]):
        if 1 in group[target].values:
            total += 1
            actual = group[group[target] == 1].iloc[0]["DriverId"]
            pred = group.sort_values(by="Prob", ascending=False).iloc[0][
                "DriverId"
            ]
            if actual == pred:
                correct += 1

    acc = (correct / total * 100) if total > 0 else 0
    print(f"\n دقة التنبؤ بالفائز الفعلي بالسباق: {acc:.1f}% ({correct}/{total})")

    # حفظ آخر حالة إحصائية للسائقين والفرق لاستخدامها في الداشبورد
    latest_driver = (
        df.groupby("DriverId")["Position"]
        .tail(3)
        .groupby(df["DriverId"])
        .mean()
        .to_dict()
    )
    latest_team = (
        df.groupby("TeamName")["Position"]
        .tail(3)
        .groupby(df["TeamName"])
        .mean()
        .to_dict()
    )

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "driver_stats": latest_driver,
            "team_stats": latest_team,
        },
        MODEL_SAVE_PATH,
    )
    print(f" تم حفظ المودل والإحصائيات في: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_and_evaluate()