import datetime
import os
import fastf1
import pandas as pd

# تحديد المسارات بالنسبة لمجلد المشروع الرئيسي
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "f1_raw_data.csv")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def fetch_season_data(year: int) -> pd.DataFrame:
    print(f"\n--- جاري فحص وسحب موسم {year} ---")
    schedule = fastf1.get_event_schedule(year)
    race_events = schedule[schedule["EventFormat"] != "testing"]

    records = []
    current_time = datetime.datetime.now(datetime.timezone.utc)

    for _, event in race_events.iterrows():
        round_num = event["RoundNumber"]
        event_name = event["EventName"]

        # تخطي السباقات التي لم تُقم بعد
        session_date = pd.to_datetime(event["EventDate"]).tz_localize("UTC")
        if session_date > current_time:
            continue

        try:
            # 1. نتائج السباق
            race = fastf1.get_session(year, round_num, "R")
            race.load(laps=False, telemetry=False, weather=False)
            r_df = race.results.copy()

            # 2. نتائج التأهيل
            quali = fastf1.get_session(year, round_num, "Q")
            quali.load(laps=False, telemetry=False, weather=False)
            q_df = quali.results.copy()

            # تحويل التوقيت إلى ثوانٍ وحساب الفارق
            for col in ["Q1", "Q2", "Q3"]:
                q_df[f"{col}_sec"] = pd.to_timedelta(q_df[col]).dt.total_seconds()

            q_df["BestQualiTime"] = q_df[["Q1_sec", "Q2_sec", "Q3_sec"]].min(
                axis=1
            )
            pole_time = q_df["BestQualiTime"].min()
            q_df["DeltaToPole"] = (q_df["BestQualiTime"] - pole_time).fillna(5.0)

            q_subset = q_df[["DriverId", "Position", "DeltaToPole"]].rename(
                columns={"Position": "Q_Position"}
            )

            # دمج التأهيل مع السباق
            merged = pd.merge(r_df, q_subset, on="DriverId", how="left")
            merged["Year"] = year
            merged["RoundNumber"] = round_num
            merged["EventName"] = event_name

            records.append(merged)
            print(f" تم سحب: جولة {round_num} - {event_name}")

        except Exception as e:
            print(f" تعذر سحب جولة {event_name}: {e}")

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def main():
    years = [2024, 2025, 2026]
    all_dfs = [fetch_season_data(y) for y in years]
    full_df = pd.concat([d for d in all_dfs if not d.empty], ignore_index=True)

    full_df.to_csv(RAW_DATA_PATH, index=False)
    print(f"\n اكتمل السحب! تم حفظ البيانات في: {RAW_DATA_PATH}")


if __name__ == "__main__":
    main()