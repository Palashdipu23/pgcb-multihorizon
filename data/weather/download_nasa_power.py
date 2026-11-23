"""
Download hourly NASA POWER weather for the eight divisional points used in the paper.

Variables (7 per station): T2M, RH2M, WS10M, PS, PRECTOTCORR, PSC, WSC
Stations and coordinates: data/stations.csv
Source: https://power.larc.nasa.gov  (meteorology from MERRA-2, 0.5 deg x 0.625 deg grid)

    python data/weather/download_nasa_power.py --start 20151028 --end 20260310

Writes data/weather/raw/<STATION>.csv and data/weather/nasa_power_hourly_wide.csv with one column
per variable and station (e.g. T2M_DHA), which is the naming used in the model input csv.
"""
import argparse
import io
import os
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"
VARS = ["T2M", "RH2M", "WS10M", "PS", "PRECTOTCORR", "PSC", "WSC"]


def fetch(lat, lon, start, end, community, time_standard):
    params = {"parameters": ",".join(VARS), "community": community, "latitude": lat, "longitude": lon,
              "start": start, "end": end, "format": "CSV", "time-standard": time_standard}
    r = requests.get(URL, params=params, timeout=300)
    r.raise_for_status()
    text = r.text
    body = text.split("-END HEADER-", 1)[1] if "-END HEADER-" in text else text
    df = pd.read_csv(io.StringIO(body.strip()))
    df["datetime"] = pd.to_datetime(dict(year=df["YEAR"], month=df["MO"], day=df["DY"], hour=df["HR"]))
    return df.drop(columns=["YEAR", "MO", "DY", "HR"]).replace(-999.0, float("nan"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="20151028")
    p.add_argument("--end", default="20260310")
    p.add_argument("--community", default="RE")
    p.add_argument("--time-standard", default="LST", choices=["LST", "UTC"],
                   help="LST = local solar time; use the same choice as the demand timestamps")
    a = p.parse_args()
    st = pd.read_csv(os.path.join(HERE, "..", "stations.csv"))
    os.makedirs(os.path.join(HERE, "raw"), exist_ok=True)
    wide = None
    for _, s in st.iterrows():
        print(f"{s.station} ({s.division}) {s.lat}, {s.lon} ...")
        df = fetch(s.lat, s.lon, a.start, a.end, a.community, a.time_standard)
        df.to_csv(os.path.join(HERE, "raw", f"{s.station}.csv"), index=False)
        df = df.set_index("datetime").rename(columns={v: f"{v}_{s.station}" for v in VARS})
        wide = df if wide is None else wide.join(df, how="outer")
        time.sleep(1)
    out = os.path.join(HERE, "nasa_power_hourly_wide.csv")
    wide.reset_index().to_csv(out, index=False)
    print("saved", out, wide.shape)


if __name__ == "__main__":
    main()
