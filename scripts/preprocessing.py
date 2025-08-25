import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import QuantileTransformer, StandardScaler, PowerTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor

def feature_engineering(df):
    df = df.copy()
    df["close_pct_change"] = df["close"].pct_change()
    df["momentum"] = df["close"] / df["close"].shift(3)
    df["cum_return_7"] = df["close"].pct_change().rolling(7).sum()
    df["sma_10"] = df["close"].rolling(window=10).mean()
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_ratio_10_20"] = df["sma_10"] / df["sma_20"]
    df["log_return_5"] = np.log(df["close"] / df["close"].shift(5))
    df["high_low"] = df["high"] - df["low"]
    df["low_close"] = (df["low"] - df["close"].shift()).abs()
    df["candle_body_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"] + 1e-8)
    df['close_pct_change_lag1'] = df['close_pct_change'].shift(1)
    df['momentum_lag1'] = df['momentum'].shift(1)
    df['sma_ratio_10_20_lag1'] = df['sma_ratio_10_20'].shift(1)
    df['cum_return_7_lag1'] = df['cum_return_7'].shift(1)
    return df[[
        "close_pct_change", 'close_pct_change_lag1',
        'cum_return_7_lag1', 'low_close', 'high_low', 
        'sma_ratio_10_20_lag1', "log_return_5", 
        'candle_body_ratio', 'momentum_lag1'
    ]]

def analyze_features(X, y, transformer, transformer_name=""):
    """
    Applies a transformer to X and returns skew, kurtosis, MI, and VIF.

    Parameters:
    -----------
    X : pd.DataFrame
        Original features.
    y : pd.Series or array
        Binary target.
    transformer : sklearn Transformer instance
        E.g., QuantileTransformer, StandardScaler, etc.
    transformer_name : str
        Optional name of the transformer (for printing).

    Returns:
    --------
    dict with:
        - skewness: pd.Series
        - kurtosis: pd.Series
        - mutual_info: pd.Series
        - vif: pd.DataFrame with columns ['feature', 'VIF']
    """
    # Fit and transform
    X_scaled = transformer.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    # Skew and kurtosis
    skewness = X_scaled_df.skew()
    kurtosis = X_scaled_df.kurtosis()

    # Mutual Information
    mi = mutual_info_classif(X_scaled_df, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    # VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled_df.values, i)
                       for i in range(X_scaled_df.shape[1])]

    print(f"\n==== Análisis con {transformer_name or transformer.__class__.__name__} ====\n")
    print(">> Skew\n", skewness.round(3))
    print("\n>> Kurtosis\n", kurtosis.round(3))
    print("\n>> Mutual Information\n", mi_series.round(3))
    print("\n>> VIF\n", vif_data.sort_values("VIF", ascending=False).round(3))

    return {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "mutual_info": mi_series,
        "vif": vif_data.sort_values("VIF", ascending=False)
    }