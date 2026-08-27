"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
QuickCart Logistics Analytics

Trains and evaluates Linear Regression and Random Forest models to forecast
shipment delivery time, then generates the model diagnostic charts used in
the Week 4 report.

Run:
    python src/week4_predictive_modeling.py
Input:
    data/cleaned_dataset.csv
Output:
    outputs/charts/chart6_actual_vs_predicted.png
    outputs/charts/chart7_feature_importance.png
    outputs/model_results.json
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

ORANGE = "#F5820D"
NAVY = "#1F2A44"
OUT_CHARTS = "outputs/charts"
OUT_TABLES = "outputs"


def metrics(y_true, y_pred):
    return {
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 3),
    }


def train_and_evaluate(df: pd.DataFrame):
    os.makedirs(OUT_CHARTS, exist_ok=True)

    model_df = pd.get_dummies(df, columns=["shipping_mode", "warehouse"], drop_first=True)
    feature_cols = [c for c in model_df.columns if c not in
                    ["order_id", "actual_days", "promised_days", "delayed", "customer_rating"]]
    X, y = model_df[feature_cols], model_df["actual_days"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Baseline
    lin = LinearRegression().fit(X_train, y_train)
    lin_pred = lin.predict(X_test)

    # Tuned Random Forest
    grid = GridSearchCV(
        RandomForestRegressor(random_state=42),
        {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
        cv=3, scoring="neg_mean_squared_error",
    )
    grid.fit(X_train, y_train)
    rf = grid.best_estimator_
    rf_pred = rf.predict(X_test)

    results = {
        "LinearRegression": metrics(y_test, lin_pred),
        "RandomForest": metrics(y_test, rf_pred),
        "best_rf_params": grid.best_params_,
    }
    with open(f"{OUT_TABLES}/model_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # Actual vs predicted (Random Forest)
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(y_test, rf_pred, alpha=0.5, color=ORANGE, s=20)
    lims = [min(y_test.min(), rf_pred.min()), max(y_test.max(), rf_pred.max())]
    plt.plot(lims, lims, "--", color=NAVY)
    plt.xlabel("Actual Delivery Time (days)"); plt.ylabel("Predicted Delivery Time (days)")
    plt.title("Random Forest: Actual vs Predicted Delivery Time")
    plt.tight_layout(); plt.savefig(f"{OUT_CHARTS}/chart6_actual_vs_predicted.png", dpi=150); plt.close()

    # Feature importance
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(8)
    plt.figure(figsize=(6.5, 4))
    importances.iloc[::-1].plot(kind="barh", color=ORANGE)
    plt.title("Top Feature Importances - Random Forest Model")
    plt.xlabel("Relative Importance")
    plt.tight_layout(); plt.savefig(f"{OUT_CHARTS}/chart7_feature_importance.png", dpi=150); plt.close()

    return results


if __name__ == "__main__":
    cleaned = pd.read_csv("data/cleaned_dataset.csv")
    train_and_evaluate(cleaned)
