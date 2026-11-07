"""Clean complaints.csv (5.7 GB) in chunks and save the result as complaints_clean.csv.

Run once:  .venv/bin/python clean.py
Output:    complaints_clean.csv  (same CSV format as the original, cleaned)
"""
import os

import pandas as pd

SRC = "complaints.csv"
DST = "complaints_clean.csv"
CHUNK = 1_000_000

# Columns we keep. "Company public response" (~70% empty, boilerplate text)
# and "ZIP code" are dropped to save memory.
USECOLS = [
    "Date received", "Product", "Sub-product", "Issue", "Sub-issue",
    "Company", "State", "Tags", "Submitted via", "Date sent to company",
    "Company response to consumer", "Timely response?", "Complaint ID",
]

# The CFPB renamed several products over the years -> merge old names into new ones
PRODUCT_MAP = {
    "Credit reporting, credit repair services, or other personal consumer reports":
        "Credit reporting or other personal consumer reports",
    "Credit reporting": "Credit reporting or other personal consumer reports",
    "Credit card or prepaid card": "Credit card",
    "Bank account or service": "Checking or savings account",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
    "Payday loan": "Payday loan, title loan, personal loan, or advance loan",
    "Payday loan, title loan, or personal loan":
        "Payday loan, title loan, personal loan, or advance loan",
    "Consumer Loan": "Vehicle loan or lease",
    "Other financial service": "Debt or credit management",
}

# Short, readable labels for the plots
SHORT = {
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Payday loan, title loan, personal loan, or advance loan": "Payday / personal loan",
}

RENAME = {
    "Date received": "date_received", "Product": "product", "Sub-product": "sub_product",
    "Issue": "issue", "Sub-issue": "sub_issue", "Company": "company", "State": "state",
    "Tags": "tags", "Submitted via": "channel", "Date sent to company": "date_sent",
    "Company response to consumer": "response", "Timely response?": "timely",
    "Complaint ID": "complaint_id",
}


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME)

    # Dates -> datetime, drop rows with unreadable dates
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce")
    df["date_sent"] = pd.to_datetime(df["date_sent"], errors="coerce")
    df = df.dropna(subset=["date_received", "product", "company"])

    # Text cleanup
    for col in ["product", "sub_product", "issue", "sub_issue", "company",
                "state", "tags", "channel", "response"]:
        df[col] = df[col].str.strip()

    df["product"] = df["product"].replace(PRODUCT_MAP).replace(SHORT)
    df["company"] = df["company"].str.upper()
    df["state"] = df["state"].str.upper()
    df["tags"] = df["tags"].fillna("None")
    df["sub_issue"] = df["sub_issue"].fillna("Unknown")
    df["sub_product"] = df["sub_product"].fillna("Unknown")

    # Yes/No -> bool
    df["timely"] = df["timely"].eq("Yes")

    # New useful columns
    df["days_to_company"] = (df["date_sent"] - df["date_received"]).dt.days.astype("Int16")
    df["year"] = df["date_received"].dt.year.astype("int16")
    df["complaint_id"] = pd.to_numeric(df["complaint_id"], errors="coerce").astype("Int64")
    return df


def main():
    total_in = total_out = 0
    seen_ids = set()  # complaint IDs already written (dedupe across chunks)

    if os.path.exists(DST):
        os.remove(DST)

    reader = pd.read_csv(SRC, usecols=USECOLS, dtype=str, chunksize=CHUNK)
    for i, chunk in enumerate(reader, 1):
        total_in += len(chunk)
        df = clean_chunk(chunk)

        # Remove duplicate complaints (inside this chunk and vs. earlier chunks)
        df = df.drop_duplicates(subset="complaint_id")
        df = df[~df["complaint_id"].isin(seen_ids)]
        seen_ids.update(df["complaint_id"].dropna().tolist())

        # Append to the CSV; write the header only for the first chunk
        df.to_csv(DST, mode="a", index=False, header=(i == 1),
                  date_format="%Y-%m-%d")
        total_out += len(df)
        print(f"chunk {i}: {total_in:,} rows read, {total_out:,} kept", flush=True)

    print(f"\nDone -> {DST}  ({total_out:,} rows, {total_in - total_out:,} removed)")


if __name__ == "__main__":
    main()
