import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "f1_raw_data.csv")
PROCESSED_DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "f1_processed_data.csv"
)


def build_features():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"الملف غير موجود: {RAW_DATA_PATH}. شغّل data_loader.py أولاً."
        )

    df = pd.read_csv(RAW_DATA_PATH)

    # 1. تنظيف القيم الأساسية
    df["GridPosition"] = (
        pd.to_numeric(df["GridPosition"], errors="coerce")
        .replace(0, 20)
        .fillna(20)
        .astype(int)
    )
    df["Position"] = (
        pd.to_numeric(df["Position"], errors="coerce").fillna(20).astype(int)
    )
    df["Q_Position"] = df["Q_Position"].fillna(20).astype(int)
    df["DeltaToPole"] = df["DeltaToPole"].fillna(5.0)
    df["IsWinner"] = (df["Position"] == 1).astype(int)

    # 2. ترتيب زمني صارم
    df = df.sort_values(by=["Year", "RoundNumber", "DriverId"]).reset_index(
        drop=True
    )

    # 3. حساب متوسط السائق في آخر 3 سباقات (منع تسريب السباق الحالي عبر shift)
    df["Driver_Avg_Finish_Last3"] = (
        df.groupby("DriverId")["Position"].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
    ).fillna(df["GridPosition"])

    # 4. حساب أداء الفريق في آخر 3 سباقات
    team_pos = (
        df.groupby(["Year", "RoundNumber", "TeamName"])["Position"]
        .mean()
        .reset_index()
    )
    team_pos = team_pos.sort_values(by=["Year", "RoundNumber"]).reset_index(
        drop=True
    )
    team_pos["Team_Avg_Finish_Last3"] = team_pos.groupby("TeamName")[
        "Position"
    ].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())

    df = pd.merge(
        df,
        team_pos[
            ["Year", "RoundNumber", "TeamName", "Team_Avg_Finish_Last3"]
        ],
        on=["Year", "RoundNumber", "TeamName"],
        how="left",
    )
    df["Team_Avg_Finish_Last3"] = df["Team_Avg_Finish_Last3"].fillna(
        df["GridPosition"]
    )

    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f" اكتملت هندسة الخصائص! تم الحفظ في: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    build_features()