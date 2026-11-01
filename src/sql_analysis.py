"""
Loads the cleaned, store-merged Rossmann dataset into an in-memory SQLite
database and runs the analytical queries in sql/rossmann_analysis.sql,
saving each result set to reports/ as a CSV.

The Python pipeline (src/data_pipeline.py) handles cleaning, feature
engineering, and sequence building for the RNN models; this script covers
the exploratory, aggregate-level SQL analysis (trends, promo/holiday
impact, store rankings).
"""
import re
import sqlite3
from pathlib import Path

import pandas as pd

from data_pipeline import load_raw_data, merge_and_clean

ROOT = Path(__file__).resolve().parent.parent
SQL_FILE = ROOT / "sql" / "rossmann_analysis.sql"
REPORTS_DIR = ROOT / "reports"

QUERY_NAMES = [
    "weekly_sales_trend",
    "promo_impact_by_storetype",
    "holiday_impact",
    "day_of_week_seasonality",
    "top20_stores",
    "competition_distance_impact",
    "monthly_seasonality",
]


def split_queries(sql_text: str) -> list[str]:
    # Strip full-line comments, then split on ';'
    lines = [l for l in sql_text.splitlines() if not l.strip().startswith("--")]
    cleaned = "\n".join(lines)
    queries = [q.strip() for q in cleaned.split(";") if q.strip()]
    return queries


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    train, store = load_raw_data()
    df = merge_and_clean(train, store)
    print(f"Loaded {len(df):,} cleaned rows into SQLite (in-memory)")

    conn = sqlite3.connect(":memory:")
    df.to_sql("sales", conn, index=False, if_exists="replace")

    queries = split_queries(SQL_FILE.read_text())
    assert len(queries) == len(QUERY_NAMES), f"{len(queries)} queries found, expected {len(QUERY_NAMES)}"

    for name, q in zip(QUERY_NAMES, queries):
        result = pd.read_sql(q, conn)
        out_path = REPORTS_DIR / f"sql_{name}.csv"
        result.to_csv(out_path, index=False)
        print(f"\n=== {name} ({len(result)} rows) -> {out_path.name} ===")
        print(result.head(10).to_string(index=False))

    conn.close()


if __name__ == "__main__":
    main()
