import pandas as pd

def clean_file(path_in, path_out):
    df = pd.read_csv(path_in)

    df = df.drop_duplicates()
    df = df.dropna()
    df = df.sort_values("time")

    df.to_csv(path_out, index=False)

def run():
    clean_file("data/raw/xauusd_m5.csv", "data/clean/xauusd_m5_clean.csv")
    clean_file("data/raw/xauusd_h1.csv", "data/clean/xauusd_h1_clean.csv")

    print("Clean data saved")