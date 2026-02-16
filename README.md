# Quantitative Trading: Bitcoin Directional Forecasting Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.3-orange) ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458) ![Status](https://img.shields.io/badge/Status-Production%20Ready-green)

A rigorous machine learning framework for predicting the directional movement of high-volatility assets (Bitcoin/Crypto). 

Unlike standard forecasting projects, this repository focuses on **Data Leakage Prevention** and statistical robustness using advanced Time-Series Cross-Validation techniques found in institutional quantitative finance (e.g., *Advances in Financial Machine Learning* by Lopez de Prado).

## Key Methodologies

### 1. Leakage-Free Validation (The "Purged" Approach)
Standard K-Fold CV fails in finance due to serial correlation. This project implements **PurgedGroupTimeSeriesSplit**:
* **Purging:** Removes training samples immediately following the test set to prevent "look-ahead bias" from overlapping labels.
* **Embargo:** Adds a safety gap after test sets to eliminate leakage from serial correlation.

### 2. Feature Engineering & Selection
* **Technical Indicators:** Custom generation of RSI, MACD, and Bollinger Bands ratios.
* **Stationarity:** Transformations applied to ensure statistical properties (log-returns, differencing).
* **Pipeline:** Integrated `QuantileTransformer` and `PowerTransformer` to handle the non-normal distribution of crypto returns.

### 3. Model Calibration
The final model is an **MLPClassifier** (Multi-Layer Perceptron) optimized not just for accuracy, but for **calibration** (probability reliability).
* **Metrics:** Focused on F1-Score and Brier Score rather than raw Accuracy, prioritizing the identification of high-confidence signals.

## Results Summary

| Metric | Score (Hold-out Test) | Insight |
| :--- | :--- | :--- |
| **F1-Score** | **0.73** | High harmonic mean of precision and recall. |
| **Accuracy** | 0.58 | Consistent with market efficiency in high-frequency assets. |
| **ROC AUC** | 0.48 | Indicates difficulty in separating classes in noisy regimes. |

*Note: While predictive power is limited by market efficiency, the pipeline ensures that results are statistically valid and not the result of overfitting or leakage.*

## Installation & Usage

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Pipeline:**
    Executes data extraction, feature engineering, and model validation.
    ```bash
    python src/run_pipeline.py
    ```

3.  **Explore the Validation Logic:**
    Check `notebooks/02_Model_Validation.ipynb` for a visual demonstration of the Purged CV splits.

## Repository Structure

```bash
├── src/
│   ├── features/       # Technical indicator generation
│   ├── models/         # PurgedCV and Model Pipeline definitions
│   └── data/           # Data fetching scripts (Yahoo Finance / Binance API)
├── notebooks/          # EDA and Validation Visualizations
└── reports/            # Performance metrics and plots
```
Developed by Santiago Daleffe - Independent Data Scientist.
