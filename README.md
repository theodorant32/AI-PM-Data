# Weather Forecasting with XGBoost

A time-series forecasting project that predicts temperature 12 hours ahead using XGBoost. Built for the PM Accelerator program to demonstrate end-to-end data science: cleaning, EDA, modeling, and interpretable results.

---

## The Problem

Can we predict temperature 12 hours ahead better than just assuming "it stays the same"? 

**Answer:** Yes — 39.8% better.

---

## Dataset

**Source:** Global Weather Repository (Kaggle)

| Statistic | Value |
|-----------|-------|
| Raw records | 138,193 rows × 41 columns |
| After cleaning | 110,486 rows |
| Locations | 257 cities worldwide |
| Date range | May 2024 – April 2026 |
| Variables | Temperature, humidity, wind, pressure, air quality, moon phases |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data (if needed)
python scripts/generate_sample_data.py

# Open notebooks and run in order
jupyter notebook
```

Run the notebooks in sequence: **01 → 02 → 03 → 04**

---

## Project Structure

```
weather-forecasting/
├── data/
│   ├── GlobalWeatherRepository.csv    # Raw data
│   └── cleaned_weather.csv            # Processed data
├── notebooks/
│   ├── 01_cleaning.ipynb              # Data preparation
│   ├── 02_eda.ipynb                   # Exploratory analysis
│   ├── 03_modeling_xgboost.ipynb      # Model training & evaluation
│   └── 04_feature_analysis.ipynb      # Feature importance
├── outputs/
│   ├── plots/                         # 6 visualizations
│   └── predictions.csv                # Model predictions
├── scripts/
│   └── generate_sample_data.py
├── requirements.txt
└── README.md
```

---

## Methodology

### 1. Data Cleaning (`01_cleaning.ipynb`)

The raw data was surprisingly complete—no missing values. But I still needed to:

- **Deduplicate:** Remove 1 row with identical location + timestamp
- **Remove outliers:** Used IQR method on temperature, humidity, and precipitation (removed ~20% of extreme values)
- **Sort chronologically:** Critical for time-series—lag features only work if data is ordered

**Why median imputation?** Weather data has spikes (heatwaves, storms). Mean would skew the baseline; median is robust.

---

### 2. Exploratory Data Analysis (`02_eda.ipynb`)

Key patterns I found:

- **Temperature range:** -2.4°C to 46.3°C (truly global dataset)
- **Distribution:** Near-normal, centered at 21.6°C
- **Seasonal cycles:** Clear peaks/troughs corresponding to summer/winter
- **Precipitation:** Highly sporadic—typical weather behavior
- **Correlations:** Humidity and precipitation move together (as expected)

The EDA informed my feature engineering: I knew I needed to capture both daily cycles (hour of day) and yearly seasons (month).

---

### 3. Feature Engineering (`03_modeling_xgboost.ipynb`)

Created **25 features** in 5 categories:

| Category | Features | Why |
|----------|----------|-----|
| **Lag** | lag_1, lag_2, lag_6, lag_12, lag_24, lag_48 | Past temperatures predict future |
| **Rolling stats** | 6-step & 24-step rolling avg/std | Smoothed trends beat single readings |
| **Cyclical time** | hour_sin/cos, month_sin/cos | 11pm is close to 1am—circular encoding |
| **Trend** | lag_1 - lag_6 | Momentum direction |
| **Interactions** | feels_like_diff, humidity × precip | Domain-specific combinations |

**Target:** Temperature 24 time-steps ahead = 12 hours

**Important:** All features computed **per-location**. London's weather doesn't predict Lagos's.

---

### 4. Model Training (`03_modeling_xgboost.ipynb`)

**Model:** XGBoost Regressor

```python
XGBRegressor(
    n_estimators=300,
    max_depth=6,           # Prevents overfitting
    learning_rate=0.05,    # Slow, stable learning
    subsample=0.8,         # Row sampling
    colsample_bytree=0.8,  # Feature sampling
    reg_lambda=1.0         # L2 regularization
)
```

**Train/Test Split:** Time-based 80/20 (not random!)
- Train: July 2024 → April 2026
- Test: Held-out future period

Random splits would leak future information and inflate performance. Time-based splits reflect real-world deployment.

---

## Results

### Model Performance

| Metric | Naive Baseline | XGBoost | Improvement |
|--------|----------------|---------|-------------|
| **MAE** | 3.93°C | 2.37°C | **39.8%** |
| **RMSE** | 5.34°C | 3.14°C | **41.2%** |

**Interpretation:** On average, predictions are off by only 2.4°C. If it's 25°C now, the model predicts within roughly ±2.5°C.

### Feature Importance

Top 5 drivers:

1. **rolling_avg_6** (57.1%) — 6-step rolling average
2. **lag_1** (16.8%) — Previous reading (30 min ago)
3. **rolling_avg_24** (6.0%) — 24-step rolling average
4. **lag_2** (4.5%) — Temperature from 1 hour ago
5. **month_cos** (2.5%) — Seasonal yearly pattern

**Key insight:** Smoothed recent trends are 3× more predictive than any single reading. This is actionable: "trust the trend, not the instant."

---

## Visualizations

All plots saved to `outputs/plots/`:

| File | Description |
|------|-------------|
| `temperature_over_time.png` | Full time-series trend |
| `precipitation_over_time.png` | Rainfall patterns |
| `distributions.png` | Histograms of temp, humidity, precip |
| `correlation_heatmap.png` | Variable relationships |
| `predicted_vs_actual.png` | Model accuracy scatter plot |
| `feature_importance.png` | Feature ranking bar chart |

---

## Real-World Applications

This isn't just an academic exercise. A 40% improvement matters when:

- 🏗️ **Construction:** Schedule concrete pouring (temperature-sensitive)
- 🌾 **Agriculture:** Frost warnings, harvest timing
- ⚡ **Energy:** Load forecasting for grid management
- 📅 **Event Planning:** Outdoor event contingency decisions

---

## Limitations

Honesty about what the model *doesn't* do:

- ❌ **12-hour horizon only** — not multi-day forecasting
- ❌ **Temperature only** — doesn't predict precipitation or conditions
- ❌ **Requires recent history** — needs continuous data stream

A good PM knows when NOT to deploy as much as when to deploy.

---

## How to Reproduce

1. **Clone the repo**
   ```bash
   git clone <your-repo-url>
   cd weather-forecasting
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run notebooks in order**
   - `01_cleaning.ipynb` — Produces `cleaned_weather.csv`
   - `02_eda.ipynb` — Generates exploratory plots
   - `03_modeling_xgboost.ipynb` — Trains model, saves predictions
   - `04_feature_analysis.ipynb` — Analyzes feature importance

4. **Check outputs**
   - `outputs/plots/` — 6 visualizations
   - `outputs/predictions.csv` — 19,073 predictions with actuals

---

## Files Explained

| File | Purpose |
|------|---------|
| `01_cleaning.ipynb` | Data prep: deduplication, outlier removal, sorting |
| `02_eda.ipynb` | Visual exploration and correlation analysis |
| `03_modeling_xgboost.ipynb` | Feature engineering, training, evaluation |
| `04_feature_analysis.ipynb` | Interprets what drives predictions |
| `requirements.txt` | All Python dependencies |
| `generate_sample_data.py` | Creates sample data if you don't have the full dataset |

---

## Dependencies

See `requirements.txt`. Key packages:

- pandas, numpy — Data manipulation
- matplotlib, seaborn — Visualizations
- xgboost — ML model
- scikit-learn — Metrics and utilities

---

## Lessons Learned

1. **Time-based splits prevent data leakage** — Random splits cheat in time-series
2. **Feature engineering > hyperparameter tuning** — Good features beat fancy models
3. **Rolling statistics beat single lags** — Smoothed trends are more stable
4. **Per-location processing matters** — Don't mix climates

---

## Author

Built by [Your Name] for PM Accelerator.

**Demo Video:** [Link to your 2-min screen recording]
**Submission:** [Link to Google Form confirmation]

---

## License

Open-source for educational purposes. Feel free to fork and experiment.
