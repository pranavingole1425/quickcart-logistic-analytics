# QuickCart Logistics Analytics

A 4-week logistics data analytics project built for the **Logistics Data Analyst Intern** virtual internship (Yuva Intern x NSDC). It simulates and analyses shipment data for **QuickCart Logistics Pvt. Ltd.**, a fictional third-party logistics (3PL) provider running last-mile delivery across five Indian metro hubs (Mumbai, Pune, Delhi-NCR, Bangalore, Hyderabad).

## Project Structure

```
quickcart-logistics-analytics/
├── QuickCart_Logistics_Analytics.ipynb   # full pipeline in one executed Jupyter notebook
├── data/
│   ├── raw_dataset.csv          # simulated raw shipment data (with data-quality issues)
│   └── cleaned_dataset.csv      # cleaned, analysis-ready dataset
├── src/
│   ├── week1_generate_data.py       # Week 1 - dataset simulation
│   ├── week2_clean_data.py          # Week 2 - cleaning & preprocessing
│   ├── week3_eda_visualization.py   # Week 3 - EDA & visualizations
│   └── week4_predictive_modeling.py # Week 4 - modeling & optimization
├── outputs/
│   ├── charts/                  # generated PNG charts (7 figures)
│   ├── eda_summary.csv
│   ├── correlation_matrix.csv
│   ├── otd_by_warehouse.csv
│   └── model_results.json
├── reports/                     # weekly DOCX submission reports
├── requirements.txt
└── README.md
```

## Key Performance Indicators

- **On-Time Delivery Rate (OTD %)**
- **Average Delivery Time (ADT)**
- **Cost per Shipment (₹)**
- **Order Fulfilment Accuracy**

## Pipeline

| Week | Script | Description |
|------|--------|-------------|
| 1 | `week1_generate_data.py` | Simulates a realistic 1,200-row shipment dataset with intentional data-quality issues |
| 2 | `week2_clean_data.py` | Handles missing values, duplicates, inconsistent labels, and outliers (IQR winsorization); scales features |
| 3 | `week3_eda_visualization.py` | Runs EDA and produces 5 visualizations (distribution, boxplot, correlation heatmap, warehouse OTD ranking, distance-vs-time scatter) |
| 4 | `week4_predictive_modeling.py` | Trains Linear Regression and tuned Random Forest models to forecast delivery time; produces diagnostic charts |

## Results

- Both models explain **~82% of the variance (R² ≈ 0.82)** in delivery time with a **mean absolute error under 0.35 days**.
- **Distance** and **shipping mode** are the dominant drivers of both cost and delivery time — package weight has negligible impact.
- **Delhi-NCR** has the lowest on-time delivery rate (54.2%) in the network, a clear target for operational improvement.

## Setup

**Option A — run the notebook (recommended):**
```bash
pip install -r requirements.txt
jupyter notebook QuickCart_Logistics_Analytics.ipynb
```
The notebook contains the entire pipeline (all 4 weeks) in one place, already executed with all outputs and charts visible.

**Option B — run the standalone weekly scripts:**
```bash
pip install -r requirements.txt
python src/week1_generate_data.py
python src/week2_clean_data.py
python src/week3_eda_visualization.py
python src/week4_predictive_modeling.py
```

## Author

Pranav Ingole — B.Tech CSE (Data Science), G.H. Raisoni College of Engineering & Management, Amravati
