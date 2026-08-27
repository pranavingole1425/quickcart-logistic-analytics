"""
Week 1 - Strategic Planning and Data Simulation
QuickCart Logistics Analytics

Generates a realistic synthetic shipment-level dataset for QuickCart Logistics
Pvt. Ltd. (fictional 3PL), modeled on public logistics datasets referenced in
the Week 1 report (DataCo Smart Supply Chain, USAID Supply Chain Shipment
Pricing, UCI Online Retail).

Run:
    python src/week1_generate_data.py
Output:
    data/raw_dataset.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)

N = 1200
WAREHOUSES = ["Mumbai", "Pune", "Delhi-NCR", "Bangalore", "Hyderabad"]
WAREHOUSE_SPEED = {"Mumbai": 1.0, "Pune": 1.05, "Delhi-NCR": 1.15, "Bangalore": 0.95, "Hyderabad": 1.1}
MODES = ["Road", "Air", "Rail"]
MODE_PROBS = [0.62, 0.23, 0.15]
MODE_SPEED_FACTOR = {"Road": 1.15, "Air": 0.55, "Rail": 1.35}
MODE_COST_FACTOR = {"Road": 8.5, "Air": 22.0, "Rail": 6.0}


def generate_raw_dataset(n=N):
    warehouse_choice = np.random.choice(WAREHOUSES, n)
    mode_choice = np.random.choice(MODES, n, p=MODE_PROBS)
    distance_km = np.round(np.random.uniform(8, 1400, n), 1)
    weight_kg = np.round(np.random.lognormal(mean=1.1, sigma=0.6, size=n), 2)

    base_days = 0.5 + distance_km / 550.0
    speed_adj = (np.array([MODE_SPEED_FACTOR[m] for m in mode_choice])
                 * np.array([WAREHOUSE_SPEED[w] for w in warehouse_choice]))
    noise = np.random.normal(0, 0.4, n)
    actual_days = np.clip(base_days * speed_adj + noise, 0.3, None)
    promised_days = np.clip(np.round(base_days * 1.2 + np.random.normal(0.2, 0.15, n), 1), 0.5, None)

    cost_inr = (distance_km * np.array([MODE_COST_FACTOR[m] for m in mode_choice]) * 0.9
                + weight_kg * 18 + np.random.normal(0, 60, n))
    cost_inr = np.round(np.clip(cost_inr, 40, None), 2)

    delayed = (actual_days > promised_days).astype(int)
    rating = np.clip(np.round(np.random.normal(4.4 - delayed * 1.1, 0.5, n), 1), 1, 5)

    df = pd.DataFrame({
        "order_id": [f"QC{100000 + i}" for i in range(n)],
        "warehouse": warehouse_choice,
        "shipping_mode": mode_choice,
        "distance_km": distance_km,
        "weight_kg": weight_kg,
        "promised_days": promised_days,
        "actual_days": np.round(actual_days, 2),
        "cost_inr": cost_inr,
        "delayed": delayed,
        "customer_rating": rating,
    })

    # --- inject realistic data-quality issues (documented in Week 2) ---
    df.loc[np.random.choice(n, 45, replace=False), "weight_kg"] = np.nan
    df.loc[np.random.choice(n, 30, replace=False), "customer_rating"] = np.nan
    mess_idx = np.random.choice(n, 60, replace=False)
    df.loc[mess_idx, "shipping_mode"] = df.loc[mess_idx, "shipping_mode"].str.upper()
    out_idx = np.random.choice(n, 12, replace=False)
    df.loc[out_idx, "cost_inr"] = df.loc[out_idx, "cost_inr"] * np.random.uniform(4, 7, 12)

    dup_rows = df.sample(15, random_state=1)
    df_raw = pd.concat([df, dup_rows], ignore_index=True)
    return df_raw


if __name__ == "__main__":
    raw = generate_raw_dataset()
    raw.to_csv("data/raw_dataset.csv", index=False)
    print(f"Generated {len(raw)} raw shipment records -> data/raw_dataset.csv")
    print(raw.head())
