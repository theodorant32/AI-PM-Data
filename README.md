# Weather Forecasting with XGBoost

Predicts temperature 12 hours ahead using XGBoost. Built for the PM Accelerator Data Scientist assessment to demonstrate end-to-end data science: cleaning, EDA, modeling, and interpretable results.

## The Problem

Can we predict temperature 12 hours ahead better than just assuming "it stays the same"?

**Answer:** Yes - 39.8% better.

## Dataset

**Source:** Global Weather Repository (Kaggle)

| Statistic | Value |
|-----------|-------|
| Raw records | 138,193 rows × 41 columns |
| After cleaning | 110,486 rows |
| Locations | 257 cities worldwide |
| Date range | May 2024 – April 2026 |

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order: **01 → 02 → 03 → 04**

## Project Structure

```
├── data/                          # Raw and cleaned data
├── notebooks/
│   ├── 01_cleaning.ipynb          # Data preparation
│   ├── 02_eda.ipynb               # Exploratory analysis
│   ├── 03_modeling_xgboost.ipynb  # Model training & evaluation
│   └── 04_feature_analysis.ipynb  # Feature importance
├── outputs/plots/                 # 6 visualizations
├── requirements.txt
└── README.md
```

## Methodology

### 1. Data Cleaning
- Removed duplicates and extreme outliers using IQR method
- Sorted chronologically (critical for time-series lag features)
- Used median imputation—robust to weather extremes

### 2. EDA
- Temperature range: -2.4°C to 46.3°C (global dataset)
- Near-normal distribution centered at 21.6°C
- Clear seasonal cycles and humidity-precipitation correlation

### 3. Feature Engineering (25 features)
- **Lag features:** Past temperatures (1–48 steps back)
- **Rolling statistics:** 6-step and 24-step moving averages
- **Cyclical encoding:** sin/cos transforms for hour and month
- **Interactions:** feels_like_diff, humidity × precipitation

All features computed per-location—no cross-city contamination.

### 4. Model
XGBoost Regressor with time-based 80/20 train/test split (not random—prevents data leakage).

## Results

| Metric | Naive Baseline | XGBoost | Improvement |
|--------|----------------|---------|-------------|
| MAE    | 3.93°C         | 2.37°C  | **39.8%**   |
| RMSE   | 5.34°C         | 3.14°C  | **41.2%**   |

**Interpretation:** On average, predictions are off by only 2.4°C.

### Top 5 Features
1. rolling_avg_6 (57.1%) — smoothed recent trend
2. lag_1 (16.8%) — previous reading
3. rolling_avg_24 (6.0%) — 12-hour trend
4. lag_2 (4.5%) — 1 hour ago
5. month_cos (2.5%) — seasonal pattern

**Insight:** Smoothed trends are 3× more predictive than single readings.

## Visualizations

All saved to `outputs/plots/`:
- `temperature_over_time.png` — Full time-series
- `distributions.png` — Histograms
- `correlation_heatmap.png` — Variable relationships
- `predicted_vs_actual.png` — Model accuracy
- `feature_importance.png` — Feature ranking

## Real-World Applications

- 🏗️ Construction: Schedule temperature-sensitive work
- 🌾 Agriculture: Frost warnings
- ⚡ Energy: Load forecasting
- 📅 Events: Outdoor planning

## Limitations

- 12-hour horizon only 
- Temperature only
- Requires continuous historical data

## Dependencies

pandas, numpy, matplotlib, seaborn, xgboost, scikit-learn

See `requirements.txt` for full list.
