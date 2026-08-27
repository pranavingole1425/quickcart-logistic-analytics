"""
Week 3 - Advanced Data Analysis and Visualization
QuickCart Logistics Analytics

Runs exploratory data analysis on the cleaned dataset and generates the five
visualizations used in the Week 3 report.

Run:
    python src/week3_eda_visualization.py
Input:
    data/cleaned_dataset.csv
Output:
    outputs/charts/chart1_delivery_time_hist.png
    outputs/charts/chart2_cost_boxplot.png
    outputs/charts/chart3_correlation_heatmap.png
    outputs/charts/chart4_otd_by_warehouse.png
    outputs/charts/chart5_distance_vs_time_scatter.png
    outputs/eda_summary.csv
    outputs/correlation_matrix.csv
    outputs/otd_by_warehouse.csv
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ORANGE = "#F5820D"
NAVY = "#1F2A44"
OUT_CHARTS = "outputs/charts"
OUT_TABLES = "outputs"


def run_eda(df: pd.DataFrame):
    os.makedirs(OUT_CHARTS, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --- Central tendency & spread ---
    desc = df[["distance_km", "weight_kg", "actual_days", "cost_inr", "customer_rating"]].describe().round(2)
    desc.to_csv(f"{OUT_TABLES}/eda_summary.csv")
    print(desc)

    # --- Distribution of delivery time ---
    plt.figure(figsize=(6.5, 4))
    sns.histplot(df["actual_days"], bins=30, color=ORANGE, kde=True)
    plt.title("Distribution of Actual Delivery Time")
    plt.xlabel("Delivery Time (days)"); plt.ylabel("Number of Shipments")
    plt.tight_layout(); plt.savefig(f"{OUT_CHARTS}/chart1_delivery_time_hist.png", dpi=150); plt.close()

    # --- Cost by shipping mode ---
    plt.figure(figsize=(6.5, 4))
    sns.boxplot(data=df, x="shipping_mode", y="cost_inr", hue="shipping_mode",
                palette=[ORANGE, NAVY, "#8FA6C7"], legend=False)
    plt.title("Shipment Cost Distribution by Shipping Mode")
    plt.xlabel("Shipping Mode"); plt.ylabel("Cost (INR)")
    plt.tight_layout(); plt.savefig(f"{OUT_CHARTS}/chart2_cost_boxplot.png", dpi=150); plt.close()

    # --- Correlation heatmap ---
    corr = df[["distance_km", "weight_kg", "actual_days", "cost_inr", "customer_rating"]].corr().round(2)
    corr.to_csv(f"{OUT_TABLES}/correlation_matrix.csv")
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(corr, annot=True, cmap="Oranges", vmin=-1, vmax=1)
    plt.title("Correlation Matrix - Key Logistics Variables")
    plt.tight_layout(); plt.savefig(f"{OUT_CHARTS}/chart3_correlation_heatmap.png", dpi=150); plt.close()

    # --- On-time delivery rate by warehouse ---
    otd = df.assign(on_time=(df["actual_days"] <= df["promised_days"]).astype(int))
    otd_by_wh = otd.groupby("warehouse")["on_time"].mean().sort_values(ascending=False) * 100
    otd_by_wh.round(1).to_csv(f"{OUT_TABLES}/otd_by_warehouse.csv")
    plt.figure(figsize=(6.5, 4))
    otd_by_wh.plot(kind="bar", color=ORANGE)
    plt.title("On-Time Delivery Rate by Warehouse")
    plt.xlabel("Warehouse"); plt.ylabel("On-Time Delivery Rate (%)")
    plt.xticks(rotation=20); plt.tight_layout()
    plt.savefig(f"{OUT_CHARTS}/chart4_otd_by_warehouse.png", dpi=150); plt.close()

    # --- Distance vs delivery time by mode ---
    plt.figure(figsize=(6.5, 4))
    sns.scatterplot(data=df, x="distance_km", y="actual_days", hue="shipping_mode",
                     palette=[ORANGE, NAVY, "#8FA6C7"], alpha=0.6, s=25)
    plt.title("Distance vs Delivery Time by Shipping Mode")
    plt.xlabel("Distance (km)"); plt.ylabel("Delivery Time (days)")
    plt.tight_layout(); plt.savefig(f"{OUT_CHARTS}/chart5_distance_vs_time_scatter.png", dpi=150); plt.close()

    print("On-time delivery rate by warehouse:\n", otd_by_wh.round(1))
    print("\nCorrelation matrix:\n", corr)
    print("\nCharts written to", OUT_CHARTS)


if __name__ == "__main__":
    cleaned = pd.read_csv("data/cleaned_dataset.csv")
    run_eda(cleaned)
