"""Import the UCI benchmark datasets into one DuckDB file each (eval/model/<name>.duckdb).

Raw downloads live in eval/model/raw/<name>/ (UCI, CC BY 4.0). Tables keep the source columns with
snake_case names; the only reshaping is:
- adult: train + official test in one table, label '.' suffix removed, column `uci_split` kept for reference
  (no longer a benchmark task; replaced by online_shoppers)
- bike: `ts` timestamp = dteday + hr
- online_retail_ii: the flat invoice-line export normalized into invoices / invoice_lines / products /
  customers so features have to be joined (cancellations and missing customers are kept as-is)
- electricity: hourly consumption (Monash copy of UCI LD2011_2014) summed to daily per client, long format

Usage: .venv/bin/python eval/model/build_db.py [online_shoppers credit_default bike online_retail_ii electricity adult]
"""

import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


def snake(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def write(db: str, tables: dict) -> None:
    path = HERE / f"{db}.duckdb"
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    for name, df in tables.items():
        con.register("_df", df)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _df')
        con.unregister("_df")
        n = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        print(f"{db}.{name}: {n} rows, {df.shape[1]} cols")
    con.close()


def adult():
    cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
            "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
    tr = pd.read_csv(RAW / "adult/adult.data", header=None, names=cols, skipinitialspace=True)
    te = pd.read_csv(RAW / "adult/adult.test", header=None, names=cols, skipinitialspace=True, skiprows=1)
    tr["uci_split"], te["uci_split"] = "train", "test"
    df = pd.concat([tr, te], ignore_index=True)
    df["income"] = df["income"].str.rstrip(".")
    df = df[df["income"].isin([">50K", "<=50K"])]
    write("adult", {"adult": df})


def online_shoppers():
    df = pd.read_csv(RAW / "online_shoppers/online_shoppers_intention.csv")
    df.columns = [snake(c) for c in df.columns]
    write("online_shoppers", {"online_shoppers": df})


def credit_default():
    df = pd.read_excel(RAW / "credit_default/default of credit card clients.xls", header=1)
    df.columns = [snake(c) for c in df.columns]
    df = df.rename(columns={"default_payment_next_month": "default_next_month"})
    write("credit_default", {"credit_default": df})


def bike():
    df = pd.read_csv(RAW / "bike/hour.csv")
    df["dteday"] = pd.to_datetime(df["dteday"])
    df.insert(1, "ts", df["dteday"] + pd.to_timedelta(df["hr"], unit="h"))
    write("bike", {"bike_hour": df})


def online_retail_ii():
    src = next((RAW / "online_retail_ii").glob("*.xlsx"))
    sheets = pd.read_excel(src, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    raw = pd.concat(sheets.values(), ignore_index=True)
    raw.columns = [snake(c) for c in raw.columns]  # invoice, stock_code, description, quantity, invoice_date, price, customer_id, country
    raw["customer_id"] = raw["customer_id"].astype("Int64").astype(str).replace("<NA>", None)
    raw["line_no"] = raw.groupby("invoice").cumcount() + 1
    invoices = (raw.groupby("invoice", as_index=False)
                .agg(customer_id=("customer_id", "first"), invoice_date=("invoice_date", "min"), country=("country", "first")))
    lines = raw[["invoice", "line_no", "stock_code", "quantity", "price"]]
    desc = raw.dropna(subset=["description"]).groupby(["stock_code", "description"]).size().reset_index(name="n")
    products = desc.sort_values("n", ascending=False).drop_duplicates("stock_code")[["stock_code", "description"]]
    products = pd.concat([products, pd.DataFrame({"stock_code": sorted(set(lines["stock_code"]) - set(products["stock_code"]))})],
                         ignore_index=True)
    cust = raw.dropna(subset=["customer_id"]).groupby(["customer_id", "country"]).size().reset_index(name="n")
    customers = cust.sort_values("n", ascending=False).drop_duplicates("customer_id")[["customer_id", "country"]]
    write("online_retail_ii", {"invoices": invoices, "invoice_lines": lines, "products": products, "customers": customers})


def electricity():
    # The UCI download (LD2011_2014, ~250MB) kept dropping and the server does not support resume, so this uses
    # the Monash Time Series Forecasting Repository copy of the same data (Zenodo record 4656140, CC BY 4.0):
    # 321 clients, hourly, 2012-01-01 to 2014-12-31, as aggregated by Lai et al. (2017). Client names are T1..T321.
    src = RAW / "electricity_monash/electricity_hourly_dataset.tsf"
    rows = []
    in_data = False
    for line in src.read_text().splitlines():
        if line.startswith("@data"):
            in_data = True
            continue
        if not in_data or not line.strip():
            continue
        name, start, values = line.split(":", 2)
        vals = pd.to_numeric(pd.Series(values.split(",")), errors="coerce").to_numpy()
        idx = pd.date_range(pd.Timestamp(start.split(" ")[0]), periods=len(vals), freq="h")
        daily = pd.Series(vals, index=idx).resample("D").sum()
        rows.append(pd.DataFrame({"client_id": name, "day": daily.index, "kwh": daily.to_numpy()}))
    long = pd.concat(rows, ignore_index=True)
    full = long.groupby("client_id")["kwh"].apply(lambda s: (s > 0).mean() > 0.99)
    keep = set(full[full].index)
    long = long[long["client_id"].isin(keep)].reset_index(drop=True)
    print(f"electricity: kept {len(keep)} of {len(full)} clients with >99% non-zero days")
    write("electricity", {"electricity_daily": long})


if __name__ == "__main__":
    which = sys.argv[1:] or ["online_shoppers", "credit_default", "bike", "online_retail_ii", "electricity"]
    for name in which:
        globals()[name]()
