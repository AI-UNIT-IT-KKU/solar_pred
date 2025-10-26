# File: xgboost_pred.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error, median_absolute_error,
    explained_variance_score
)
from IPython.display import display
import warnings
warnings.filterwarnings("ignore")

# ================================================================
#  Main Function
# ================================================================

def xgboost_pred(master_df, param_grid, subset_fraction=0.2, train_ratio=0.8):
    """
    Runs a full XGBoost pipeline (GridSearchCV on subset, then full train)
    for a single, pre-defined feature set (G2_Smart_Choice_NonLinear).
    
    Args:
        master_df (pd.DataFrame): The main DataFrame.
        param_grid (dict): The parameter grid for GridSearchCV.
        subset_fraction (float): The fraction of data to use for fast GridSearchCV.
        train_ratio (float): The chronological train/test split ratio.

    Returns:
        tuple: (final_model, full_test_metrics, best_params, features_used, yte_f, yte_pred_f)
               Returns (None, None, None, None, None, None) on failure.
    """

    # --- 1. Define the specific features for Group 2 ---
    group_name = "G2_Smart_Choice_NonLinear"
    desired_cols = [
        "avg_isc_test_a", "avg_geff_test_w_m2",
        "avg_moduletemp1_c", "avg_temp_refcell_c",
        "avg_wind_speed_m_s", "avg_humidity_pct"
    ]

    # --- 2. Setup Data ---
    TARGET_COL = 'active_power' if 'active_power' in master_df.columns else 'activepower_1m'
    
    df = master_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.sort_index().dropna(subset=[TARGET_COL])

    # --- 3. Check features ---
    feats = _available_features(df, desired_cols)
    if len(feats) == 0:
        print(f"ERROR: {group_name}: No matching columns found in data. Skipping.")
        # 💡 تم التعديل هنا ليتوافق مع المخرجات الجديدة
        return None, None, None, None, None, None

    print("\n" + "="*70)
    print(f"--- Running Prediction for Group: {group_name} ---")
    print(f"  Using features ({len(feats)}): {feats}")

    # --- 4. Subset for GridSearchCV ---
    n_subset = max(int(len(df) * subset_fraction), 2000)
    df_sub = df.iloc[:n_subset].copy()
    
    X_sub = df_sub[feats]
    y_sub = df_sub[TARGET_COL]
    Xtr_s, Xte_s, ytr_s, yte_s = _chronological_split(X_sub, y_sub, train_ratio)

    # --- 5. Run GridSearchCV ---
    tscv = TimeSeriesSplit(n_splits=3)
    base = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
    
    print("  Running GridSearchCV on subset...")
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0
    )
    
    # ==================================================
    # Convert to simple NumPy arrays before fitting
    # This avoids the PicklingError with pandas indexes
    print("  Converting data to NumPy arrays for parallel processing...")
    X_fit = Xtr_s.values
    y_fit = ytr_s.values
    
    # Now fit using the NumPy arrays
    grid.fit(X_fit, y_fit)
    # ==================================================
    
    best_params = grid.best_params_
    print(f"  Best params (subset): {best_params}")
    print(f"  Best CV RMSE (subset): {-grid.best_score_:,.0f} W")

    # --- 6. Evaluate on Subset (for logging) ---
    best_model_subset = grid.best_estimator_
    # .predict() works fine with the original pandas DataFrame
    m_test_s = _regression_metrics(yte_s, best_model_subset.predict(Xte_s))
    _print_metrics("  Subset Test  metrics:", m_test_s)

    # --- 7. Train on Full Data ---
    print(f"\n  Training on FULL data with best params...")
    X_full = df[feats]
    y_full = df[TARGET_COL]
    Xtr_f, Xte_f, ytr_f, yte_f = _chronological_split(X_full, y_full, train_ratio)

    final_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        **best_params
    )
    # .fit() works fine with pandas DataFrames when n_jobs=1 (default)
    final_model.fit(Xtr_f, ytr_f)

    # --- 8. Evaluate on Full Data ---
    ytr_pred_f = final_model.predict(Xtr_f)
    yte_pred_f = final_model.predict(Xte_f)

    m_train_f = _regression_metrics(ytr_f, ytr_pred_f)
    m_test_f  = _regression_metrics(yte_f, yte_pred_f)

    _print_metrics("\n  FULL Train metrics:", m_train_f)
    _print_metrics("  FULL Test  metrics:",  m_test_f)
    print("="*70)
    
    # --- 9. Return artifacts ---
    return final_model, m_test_f, best_params, feats, yte_f, yte_pred_f

# ================================================================
# Helper Functions (Internal)
# ================================================================

def _available_features(df, desired_cols):
    """Return only columns that actually exist in the DataFrame."""
    return [c for c in desired_cols if c in df.columns]

def _chronological_split(X, y, train_ratio=0.8):
    """Simple chronological split (no shuffling)."""
    n = len(X)
    cut = int(n * train_ratio)
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]

def _regression_metrics(y_true, y_pred):
    """Calculate a comprehensive set of regression metrics."""
    # (Removed MAPE to avoid potential divide-by-zero on night-time data)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    return {
        "MAE": mae, "RMSE": rmse, "R2": r2, 
        "MedAE": medae, "ExplainedVar": evs
    }

def _print_metrics(title, m):
    """Prints a formatted block of regression metrics."""
    print(f"{title}")
    print(f"  MAE   = {m['MAE']:,.0f} W")
    print(f"  RMSE  = {m['RMSE']:,.0f} W")
    print(f"  R2    = {m['R2']:.4f}")
    print(f"  MedAE = {m['MedAE']:,.0f} W")
    print(f"  EVS   = {m['ExplainedVar']:.4f}")