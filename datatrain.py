import os
import requests
import pandas as pd
import time

# --- Cấu hình ---
BBOX = [8.0, 102.0, 23.5, 120.0]   # [South, West, North, East]
START_DATE = "2010-01-01"
END_DATE = "2025-09-24"
PARAMS = "T2M,QV2M,PS,WS10M,PRECTOTCORR,CLRSKY_SFC_SW_DWN"

# --- Step theo NASA POWER Grid ---
LAT_STEP = 0.5
LON_STEP = 0.625

OUTPUT_FILE = "datatrainai.csv"

# --- Hàm xác định season ---
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

# --- Hàm fetch dữ liệu 1 điểm ---
def fetch_point_data(lat, lon, start, end, params):
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters={params}&start={start}&end={end}"
        f"&latitude={lat}&longitude={lon}&community=AG&format=JSON"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    if "properties" not in data or "parameter" not in data["properties"]:
        return pd.DataFrame()

    params_dict = data["properties"]["parameter"]

    df = pd.DataFrame(params_dict)
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")

    df["Latitude"] = lat
    df["Longitude"] = lon
    df["hour"] = 0
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["season"] = df["month"].apply(get_season)

    return df

# --- Hàm float range ---
def frange(start, stop, step):
    while start <= stop:
        yield round(start, 3)
        start += step

# --- MAIN fetch ---
def fetch_bbox_data():
    years = range(int(START_DATE[:4]), int(END_DATE[:4]) + 1)

    for lat in frange(BBOX[0], BBOX[2], LAT_STEP):
        for lon in frange(BBOX[1], BBOX[3], LON_STEP):
            for year in years:
                start = f"{year}0101"
                end = f"{year}1231"
                print(f"Fetching {year} lat={lat} lon={lon} ...")

                try:
                    df = fetch_point_data(lat, lon, start, end, PARAMS)
                    if not df.empty:
                        # --- lưu ngay ---
                        if not os.path.exists(OUTPUT_FILE):
                            df.to_csv(OUTPUT_FILE, index=False, mode="w")
                        else:
                            df.to_csv(OUTPUT_FILE, index=False, mode="a", header=False)
                        print(f"✅ Saved {len(df)} rows for {year} lat={lat} lon={lon}")
                except Exception as e:
                    print("❌ Error:", e)
                time.sleep(1)  # tránh spam API

if __name__ == "__main__":
    fetch_bbox_data()
    print("🎉 Done!")