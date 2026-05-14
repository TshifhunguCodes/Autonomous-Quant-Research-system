import pandas as pd

from core.logging_utils import get_logger


logger = get_logger(__name__)


def clean_file(path_in, path_out):
    df = pd.read_csv(path_in, parse_dates=["time"])
    # Ensure consistent datetime precision
    df["time"] = df["time"].astype("datetime64[s]")
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.sort_values("time").reset_index(drop=True)
    df.to_csv(path_out, index=False)
    return len(df)


def run(config):
    m5_rows = clean_file(config.paths.raw_m5, config.paths.clean_m5)
    h1_rows = clean_file(config.paths.raw_h1, config.paths.clean_h1)
    logger.info("Clean data saved: M5 rows=%s, H1 rows=%s", m5_rows, h1_rows)
