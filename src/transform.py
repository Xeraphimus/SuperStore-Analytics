"""
Super Store - data wrangling & transformation pipeline.

Reads the raw Excel workbook (Orders / Returns / Users), cleans it, joins the
sheets, engineers reporting features and writes:

  data/processed/orders_enriched.csv      one wide, analysis-ready table
  data/processed/superstore_model.xlsx    star schema for reporting
  reports/data_quality.md                 what was found and what was done

Run from the repository root:
    python src/transform.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "super_store_us.xlsx"
OUT = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
    "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
PRIORITY_ORDER = ["Critical", "High", "Medium", "Low", "Not Specified"]


def load(path: Path = RAW) -> dict[str, pd.DataFrame]:
    return pd.read_excel(path, sheet_name=None)


def clean_orders(orders: pd.DataFrame, log: list[str]) -> pd.DataFrame:
    df = orders.copy()
    df = df.rename(columns={"Quantity ordered new": "Quantity", "State or Province": "State"})

    # 1. Stray whitespace in text columns ("Critical " vs "Critical").
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    before = df["Order Priority"].nunique()
    for c in text_cols:
        df[c] = df[c].astype("string").str.strip()
    log.append(f"Trimmed whitespace in {len(text_cols)} text columns; `Order Priority` "
               f"went from {before} to {df['Order Priority'].nunique()} distinct values.")

    # 2. Postal codes were stored as numbers, so New England zips lost their leading zero.
    short = int((df["Postal Code"] < 10000).sum())
    df["Postal Code"] = df["Postal Code"].astype(int).astype(str).str.zfill(5)
    log.append(f"Re-padded {short} postal codes to 5 digits (e.g. 7203 -> 07203).")

    # 3. Row ID should be a key but one value is used twice (two genuinely different lines).
    dup = df.loc[df["Row ID"].duplicated(keep=False), "Row ID"].unique().tolist()
    df["Line Key"] = np.arange(1, len(df) + 1)
    log.append(f"`Row ID` is not unique ({dup} appears twice on different lines); "
               "added a surrogate `Line Key` and kept both rows.")

    # 4. Product Base Margin has a few blanks - fill with the sub-category median.
    n_missing = int(df["Product Base Margin"].isna().sum())
    df["Base Margin Imputed"] = df["Product Base Margin"].isna()
    df["Product Base Margin"] = df["Product Base Margin"].fillna(
        df.groupby("Product Sub-Category")["Product Base Margin"].transform("median"))
    log.append(f"Imputed {n_missing} missing `Product Base Margin` values with the "
               "sub-category median (flagged in `Base Margin Imputed`).")

    # 5. Integrity observation: an Order ID should belong to one customer.
    per_order = df.groupby("Order ID").agg(customers=("Customer ID", "nunique"),
                                           regions=("Region", "nunique"))
    multi = per_order[per_order["customers"] > 1]
    df["Order ID Shared"] = df["Order ID"].isin(multi.index)
    log.append(f"{len(multi)} Order IDs are shared by more than one customer "
               f"({int(df['Order ID Shared'].sum())} lines); some even span regions. "
               "Kept as-is and flagged (`Order ID Shared`) - order-level counts should be read with care.")
    return df


def engineer(df: pd.DataFrame, returns: pd.DataFrame, users: pd.DataFrame,
             log: list[str]) -> pd.DataFrame:
    df = df.copy()

    # Joins -------------------------------------------------------------------
    ret = returns.assign(Status=returns["Status"].str.strip()).drop_duplicates("Order ID")
    df = df.merge(ret, on="Order ID", how="left")
    df["Returned"] = df["Status"].eq("Returned")
    df = df.drop(columns="Status")
    matched = ret["Order ID"].isin(df["Order ID"]).sum()
    log.append(f"Joined Returns on `Order ID`: only {matched} of {len(ret)} returned order IDs "
               f"exist in Orders ({int(df['Returned'].sum())} lines). Returns appears to come from a "
               "wider (global) order book, so return-rate figures are indicative only.")

    df = df.merge(users.assign(Region=users["Region"].str.strip()), on="Region", how="left")
    log.append("Joined Users on `Region` to attach the responsible regional `Manager`.")

    # Dates -------------------------------------------------------------------
    df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Order Month"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
    df["Order Month Name"] = df["Order Date"].dt.strftime("%b")
    df["Order Week"] = df["Order Date"].dt.isocalendar().week.astype(int)
    df["Order Weekday"] = df["Order Date"].dt.day_name()
    df["Order Quarter"] = "Q" + df["Order Date"].dt.quarter.astype(str)

    # Money -------------------------------------------------------------------
    df["Gross Sales"] = df["Unit Price"] * df["Quantity"]
    df["Discount Amount"] = df["Gross Sales"] * df["Discount"]
    df["Profit Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], np.nan)
    df["Is Loss"] = df["Profit"] < 0
    df["Margin Gap"] = df["Profit Margin"] - df["Product Base Margin"]

    # Bands for slicers ---------------------------------------------------------
    df["Discount Band"] = pd.cut(df["Discount"], [-0.001, 0, 0.03, 0.06, 0.09, 1],
                                 labels=["0%", "1-3%", "4-6%", "7-9%", "10%+"]).astype(str)
    df["Order Size Band"] = pd.cut(df["Sales"], [0, 100, 500, 2000, 10000, np.inf],
                                   labels=["<100", "100-500", "500-2k", "2k-10k", "10k+"]).astype(str)
    df["Ship Speed"] = pd.cut(df["Ship Days"], [-1, 1, 3, 7, np.inf],
                              labels=["Same/next day", "2-3 days", "4-7 days", "8+ days"]).astype(str)
    df["Priority Rank"] = df["Order Priority"].map({p: i for i, p in enumerate(PRIORITY_ORDER, 1)})
    df["State Code"] = df["State"].map(STATE_CODES)

    missing_codes = df.loc[df["State Code"].isna(), "State"].unique()
    if len(missing_codes):
        log.append(f"WARNING: no state code for {list(missing_codes)}")
    log.append("Engineered: ship days, calendar parts, gross sales, discount amount, profit margin, "
               "loss flag, margin gap vs. base margin, discount / order-size / ship-speed bands, "
               "priority rank and 2-letter state codes (for maps).")
    return df


def star_schema(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the wide table into a small star schema for reporting."""
    dim_customer = (df[["Customer ID", "Customer Name", "Customer Segment"]]
                    .drop_duplicates("Customer ID").sort_values("Customer ID"))

    prod_cols = ["Product Name", "Product Category", "Product Sub-Category",
                 "Product Container", "Product Base Margin"]
    dim_product = df[prod_cols].drop_duplicates("Product Name").reset_index(drop=True)
    dim_product.insert(0, "Product Key", np.arange(1, len(dim_product) + 1))

    geo_cols = ["Postal Code", "City", "State", "State Code", "Region", "Country"]
    dim_geo = df[geo_cols].drop_duplicates(["Postal Code", "City", "State"]).reset_index(drop=True)
    dim_geo.insert(0, "Geo Key", np.arange(1, len(dim_geo) + 1))

    dates = pd.date_range(df["Order Date"].min(), df["Ship Date"].max(), freq="D")
    dim_date = pd.DataFrame({"Date": dates})
    dim_date["Year"] = dim_date["Date"].dt.year
    dim_date["Quarter"] = "Q" + dim_date["Date"].dt.quarter.astype(str)
    dim_date["Month Number"] = dim_date["Date"].dt.month
    dim_date["Month"] = dim_date["Date"].dt.strftime("%b")
    dim_date["Month Start"] = dim_date["Date"].dt.to_period("M").dt.to_timestamp()
    dim_date["ISO Week"] = dim_date["Date"].dt.isocalendar().week.astype(int)
    dim_date["Weekday"] = dim_date["Date"].dt.day_name()
    dim_date["Weekday Number"] = dim_date["Date"].dt.dayofweek + 1
    dim_date["Is Weekend"] = dim_date["Weekday Number"] >= 6

    dim_region = df[["Region", "Manager"]].drop_duplicates().sort_values("Region")

    fact = (df.merge(dim_product[["Product Key", "Product Name"]], on="Product Name")
              .merge(dim_geo[["Geo Key", "Postal Code", "City", "State"]],
                     on=["Postal Code", "City", "State"]))
    fact_cols = ["Line Key", "Row ID", "Order ID", "Order Date", "Ship Date", "Customer ID",
                 "Product Key", "Geo Key", "Region", "Order Priority", "Priority Rank", "Ship Mode",
                 "Unit Price", "Quantity", "Discount", "Shipping Cost", "Gross Sales",
                 "Discount Amount", "Sales", "Profit", "Profit Margin", "Margin Gap", "Is Loss",
                 "Ship Days", "Ship Speed", "Discount Band", "Order Size Band", "Returned",
                 "Order ID Shared", "Base Margin Imputed"]
    fact = fact[fact_cols].sort_values("Line Key")
    assert len(fact) == len(df), "star-schema join changed the row count"

    return {"FactOrderLines": fact, "DimCustomer": dim_customer, "DimProduct": dim_product,
            "DimGeography": dim_geo, "DimDate": dim_date, "DimRegionManager": dim_region}


def run() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    sheets = load()
    orders, returns, users = sheets["Orders"], sheets["Returns"], sheets["Users"]
    log.append(f"Loaded Orders {orders.shape}, Returns {returns.shape}, Users {users.shape}.")

    df = engineer(clean_orders(orders, log), returns, users, log)
    df.to_csv(OUT / "orders_enriched.csv", index=False, encoding="utf-8-sig")

    tables = star_schema(df)
    with pd.ExcelWriter(OUT / "superstore_model.xlsx", engine="openpyxl") as xw:
        for name, t in tables.items():
            t.to_excel(xw, sheet_name=name, index=False)

    lines = ["# Data quality & transformation log", "",
             "Generated by `src/transform.py`.", ""]
    lines += [f"{i}. {entry}" for i, entry in enumerate(log, 1)]
    lines += ["", "## Output tables", ""]
    lines += [f"- **{n}**: {len(t):,} rows x {t.shape[1]} columns" for n, t in tables.items()]
    (REPORTS / "data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(log))
    print(f"\nWrote {len(df):,} rows -> {OUT}")
    return df


if __name__ == "__main__":
    run()
