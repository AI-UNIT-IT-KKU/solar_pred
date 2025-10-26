import pandas as pd
from typing import List
from IPython.display import display # For rich display in notebooks

def merge_files(filled_dfs_list: List[pd.DataFrame], output_path: str = "master_solar_dataset_filled.parquet") -> pd.DataFrame:
    """
    Merges a list of cleaned and filled DataFrames into a single, aggregated 
    master DataFrame.

    The process involves:
    1. Renaming columns from each file (e.g., 'col_file1', 'col_file2').
    2. Merging all DataFrames horizontally on their 'time' index.
    3. Calculating the mean for 'active_power' across all files.
    4. Calculating the mean for all other specified environmental/operational columns.
    5. Interpolating any remaining small gaps.
    6. Saving the final DataFrame to a Parquet file.

    Args:
        filled_dfs_list (List[pd.DataFrame]): The list of DataFrames to merge.
        output_path (str): The file path to save the final Parquet file.

    Returns:
        pd.DataFrame: The final, aggregated master DataFrame.
    """
    
    print("\n---  Final Step: Merging and Aggregating All Filled Files ---")

    # 🧩 Step 1: Create deep copies from the newly filled data
    # We use the passed 'filled_dfs_list'
    dfs_to_merge = [df.copy() for df in filled_dfs_list]

    #  Step 2: Ensure 'time' exists and set as index, renaming columns
    for i, df in enumerate(dfs_to_merge):
        if 'time' not in df.columns:
            print(f"⚠️ File {i+1} has no 'time' column! Skipping...")
            continue

        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])
        df = df.sort_values('time').set_index('time')

        # Rename columns to prevent conflicts (e.g., 'activepower_1m_file1')
        df.columns = [f"{col}_file{i+1}" for col in df.columns]
        dfs_to_merge[i] = df # Update the list with the processed DataFrame

    #  Step 3: Merge all files horizontally (by time)
    print(f"Merging {len(dfs_to_merge)} files horizontally...")
    wide_df = pd.concat(dfs_to_merge, axis=1, join='outer')

    #  Step 4: Create the final master dataframe
    master_df = pd.DataFrame(index=wide_df.index)

    #  Aggregate Active Power (using mean)
    power_cols = [c for c in wide_df.columns if 'activepower_1m' in c]
    if power_cols:
        master_df['active_power'] = wide_df[power_cols].mean(axis=1)

    # Step 5: Aggregate other environmental/operational columns
    # These are the *base names* of columns to average
    cols_to_average = [
        'poa1_w_m2', 'poa1_wh_m2', 'poa2_w_m2', 'poa2_wh_m2',
        'ghi_w_m2', 'ghi_wh_m2', 'ambienttemp_c', 'moduletemp1_c',
        'moduletemp2_c', 'wind_speed_m_s', 'wind_dir_deg', 'rain_mm',
        'humidity_pct', 'soiling_loss_isc_pct', 'soiling_loss_geff_pct',
        'isc_test_a', 'isc_ref_a', 'temp_test_c', 'temp_refcell_c',
        'geff_test_w_m2', 'geff_ref_w_m2'
    ]
    
    print("Averaging environmental and operational columns...")
    for base_col in cols_to_average:
        # Find all columns that match this base name (e.g., 'ghi_w_m2_file1', 'ghi_w_m2_file2')
        matches = [c for c in wide_df.columns if base_col in c]
        if matches:
            # Create a new column (e.g., 'avg_ghi_w_m2') with the mean
            master_df[f'avg_{base_col}'] = wide_df[matches].mean(axis=1)

    #  Step 6: Sort by time and interpolate small gaps
    master_df = master_df.sort_index().interpolate(method='time')

    #  Step 7: Save the final file
    try:
        master_df.to_parquet(output_path, index=True)
        print(f"\n --- Aggregation Complete! ---")
        print(f" Output saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ --- ERROR: Failed to save file! ---")
        print(f"Error: {e}")
        return master_df # Return the dataframe even if save fails

    # 📊 Step 8: Display comprehensive summary
    print(f" Shape: {master_df.shape}")
    print(f" Time Range: {master_df.index.min()} → {master_df.index.max()}")
    print(f" Columns: {list(master_df.columns)}")
    display(master_df.describe())
    
    return master_df