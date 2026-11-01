"""Download the official NASA PCoE battery archive and extract discharge capacities to CSV.

Usage:  python fetch_data.py
"""
import urllib.request

from core import NASA_CSV, NASA_ZIP, build_nasa_csv

URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"

if __name__ == "__main__":
    if not NASA_ZIP.exists():
        NASA_ZIP.parent.mkdir(exist_ok=True)
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, NASA_ZIP)
    df = build_nasa_csv()
    print(f"Wrote {NASA_CSV} ({len(df)} discharge cycles)")
    print(df.groupby("battery")["capacity_Ah"].agg(["count", "first", "last", "min"]))
