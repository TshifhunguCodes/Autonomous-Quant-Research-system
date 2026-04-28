import psycopg2
import csv

# =========================
# CONFIG
# =========================
DB_NAME = "Quant_System_Database"
DB_USER = "postgres"
DB_PASSWORD = "Tshifhungu12@"
DB_HOST = "localhost"
DB_PORT = "5432"

CSV_FILE = "data/backtest/results.csv"
TABLE_NAME = "quant_system_data"

# =========================
# CONNECT DB
# =========================
conn = psycopg2.connect(
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT
)

cur = conn.cursor()

# =========================
# READ CSV HEADER
# =========================
with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)
    headers = [h.strip().lower() for h in next(reader)]

# =========================
# COLUMN RULES
# =========================
TEXT_COLS = {
    "behavior_label", "structure_state", "pattern", "session",
    "alpha_signal", "alpha_notes", "flow_signal", "flow_notes",
    "signal", "signal_owner", "direction", "date"
}

BOOL_COLS = {
    "trend_up", "trend_down", "breakout", "reversal", "flip",
    "choppy", "bos_up", "bos_down", "bos", "choch",
    "double_top", "double_bottom", "is_support", "is_resistance",
    "supply_zone", "demand_zone", "order_block", "fvg_zone",
    "trade_allowed"
}

INT_COLS = {
    "hour", "flip_count_10", "daily_loss_locked"
}

# =========================
# CREATE TABLE
# =========================
def infer_type(col):
    if col in TEXT_COLS:
        return "TEXT"
    elif col in BOOL_COLS:
        return "INTEGER"
    elif col in INT_COLS:
        return "INTEGER"
    elif col == "time":
        return "TIMESTAMP"
    elif col == "date":
        return "DATE"
    else:
        return "DOUBLE PRECISION"

columns_sql = [
    f'"{col}" {infer_type(col)}'
    for col in headers
]

create_sql = f"""
CREATE TABLE {TABLE_NAME} (
    {", ".join(columns_sql)}
);
"""

cur.execute(create_sql)
conn.commit()

print("Table created successfully!")

# =========================
# CLEAN FUNCTION (FINAL SAFE VERSION)
# =========================
def clean_value(col, value):
    if value is None:
        return None

    value = str(value).strip()

    # NULL HANDLING
    if value == "" or value.lower() in ["nan", "none", "null"]:
        return None

    # TIME (IMPORTANT FIX)
    if col == "time":
        return value  # let PostgreSQL handle timestamp parsing

    # BOOLEAN
    if col in BOOL_COLS:
        if value.lower() == "true":
            return 1
        if value.lower() == "false":
            return 0

    # TEXT
    if col in TEXT_COLS:
        return value

    # INT
    if col in INT_COLS:
        try:
            return int(float(value))
        except:
            return None

    # FLOAT
    try:
        return float(value)
    except:
        return None

# =========================
# INSERT QUERY
# =========================
placeholders = ", ".join(["%s"] * len(headers))
columns_formatted = ", ".join([f'"{c}"' for c in headers])

insert_sql = f"""
INSERT INTO {TABLE_NAME} ({columns_formatted})
VALUES ({placeholders});
"""

# =========================
# LOAD DATA (SAFE + FAST)
# =========================
with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader)

    batch = []
    batch_size = 5000
    count = 0

    for row in reader:

        # SAFETY: avoid row mismatch
        if len(row) != len(headers):
            continue

        cleaned_row = [
            clean_value(headers[i], row[i])
            for i in range(len(headers))
        ]

        batch.append(cleaned_row)

        if len(batch) >= batch_size:
            cur.executemany(insert_sql, batch)
            conn.commit()
            count += len(batch)
            print(f"{count} rows inserted...")
            batch = []

    # remaining
    if batch:
        cur.executemany(insert_sql, batch)
        conn.commit()
        count += len(batch)

print(f"Done! {count} rows inserted successfully.")

cur.close()
conn.close()