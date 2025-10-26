# feature_selection.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from IPython.display import display

def rank_features_combined(df_normalized: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    """
    Calculates and ranks features based on a combined score of
    Pearson correlation, Spearman correlation, and LightGBM importance.
    
    Args:
        df_normalized (pd.DataFrame): The input DataFrame (e.g., df1_normalized).
        plot (bool): Whether to display the visualization.

    Returns:
        pd.DataFrame: A DataFrame with features ranked by the combined score.
    """
    
    print("--- Starting Combined Feature Ranking ---")

    # --- 1. Define Columns (as in the original script) ---
    features = ['avg_poa1_w_m2', 'avg_poa1_wh_m2', 'avg_poa2_w_m2',
       'avg_poa2_wh_m2', 'avg_ghi_w_m2', 'avg_ghi_wh_m2', 'avg_ambienttemp_c',
       'avg_moduletemp1_c', 'avg_moduletemp2_c', 'avg_wind_speed_m_s',
       'avg_wind_dir_deg', 'avg_rain_mm', 'avg_humidity_pct',
       'avg_soiling_loss_isc_pct', 'avg_soiling_loss_geff_pct',
       'avg_isc_test_a', 'avg_isc_ref_a', 'avg_temp_test_c',
       'avg_temp_refcell_c', 'avg_geff_test_w_m2', 'avg_geff_ref_w_m2']
    target = 'active_power'

    # --- 2. Prepare Data ---
    print("Preparing data...")
    df = df_normalized.copy()
    
    # Check if all required columns exist
    required_cols = features + [target]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return pd.DataFrame() # Return empty DataFrame on error
        
    df = df[required_cols].dropna()

    if df.empty:
        print("Error: DataFrame is empty after dropping NA values. Cannot proceed.")
        return pd.DataFrame()

    X = df[features]
    y = df[target]

    # ============================================================
    #  3. Pearson + Spearman Correlation
    # ============================================================
    print("Calculating correlations...")
    pearson_corr = df.corr(method='pearson')[target].drop(target)
    spearman_corr = df.corr(method='spearman')[target].drop(target)

    # ============================================================
    # 🔹 4. Feature Importance (LightGBM)
    # ============================================================
    print("Training LightGBM for feature importance...")
    # Use a time-series-aware split (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Added verbosity=-1 to suppress training output
    model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbosity=-1)
    model.fit(X_train, y_train)
    
    importance = pd.Series(model.feature_importances_, index=features)

    # ============================================================
    # 🔹 5. Combine All Metrics
    # ============================================================
    print("Combining metrics and ranking...")
    combined = pd.DataFrame({
        'Pearson': pearson_corr,
        'Spearman': spearman_corr,
        # Normalize importance to be on a 0-1 scale
        'Importance': importance / importance.max() 
    }).fillna(0) # FillNa for any potential calculation issues

    # Calculate the final score
    combined['Score'] = (abs(combined['Pearson']) + abs(combined['Spearman']) + combined['Importance']) / 3
    combined = combined.sort_values('Score', ascending=False)

    print("\n Combined Ranking of Features:")
    display(combined.round(3)) # Use display() for notebook-friendly output

    # ============================================================
    # 🔹 6. Visualization (Optional)
    # ============================================================
    if plot:
        print("Generating plot...")
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=combined.reset_index(),
            x='Score', y='index', palette='coolwarm'
        )
        plt.title(" Overall Feature Relevance (Pearson + Spearman + Importance)", fontsize=14, fontweight='bold')
        plt.xlabel("Combined Score")
        plt.ylabel("Feature")
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()
    
    print("--- Feature Ranking Complete ---")
    return combined