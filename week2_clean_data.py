"""
Week 2 - Data Collection, Cleaning, and Preprocessing
QuickCart Logistics Analytics

Cleans the raw shipment dataset: handles missing values, removes duplicates,
standardises categorical labels, caps cost outliers (IQR/winsorization), and
scales numeric features.

Run:
    python src/week2_clean_data.py
Input:
    data/raw_dataset.csv
Output:
    data/cleaned_dataset.csv
"""
import pandas as pd


def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # 1. Standardise categorical text (e.g. "ROAD" / "road" / "Road" -> "Road")
    df["shipping_mode"] = df["shipping_mode"].str.strip().str.title()

    # 2. Remove duplicate shipments, keep first occurrence
    df = df.drop_duplicates(subset="order_id", keep="first")

    # 3. Impute missing values
    df["weight_kg"] = df["weight_kg"].fillna(df["weight_kg"].median())
    df["customer_rating"] = df["customer_rating"].fillna(df["customer_rating"].mode()[0])

    # 4. Cap cost outliers using the IQR rule (winsorization, not deletion)
    q1, q3 = df["cost_inr"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["cost_inr"] = df["cost_inr"].clip(lo, hi)

    # 5. Feature scaling for modelling stage
    df["distance_norm"] = (df["distance_km"] - df["distance_km"].min()) / (
        df["distance_km"].max() - df["distance_km"].min()
    )
    df["cost_zscore"] = (df["cost_inr"] - df["cost_inr"].mean()) / df["cost_inr"].std()

    return df


if __name__ == "__main__":
    raw = pd.read_csv("data/raw_dataset.csv")
    print(f"Raw rows: {len(raw)} | missing values: {raw.isna().sum().sum()} | "
          f"duplicate order_ids: {raw.duplicated(subset='order_id').sum()}")

    cleaned = clean_dataset(raw)
    cleaned.to_csv("data/cleaned_dataset.csv", index=False)

    print(f"Cleaned rows: {len(cleaned)} | missing values: {cleaned.isna().sum().sum()}")
    print("Saved -> data/cleaned_dataset.csv")
