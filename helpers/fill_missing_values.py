import pandas as pd
import numpy as np
from typing import List


def _safe_fill_single_file(df: pd.DataFrame, days_window: int = 7) -> pd.DataFrame:
    """
    (Internal Helper) Fills missing values for a *single* DataFrame
    using a rolling daily average for the same minute.
    """
    df = df.copy()
    # 1. Prepare index
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    df = df.set_index('time').sort_index()

    # 2. Create helper columns
    df['minute_of_day'] = df.index.hour * 60 + df.index.minute
    df['date'] = df.index.date

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    filled_cols = []
    
    print(f"  > Processing {len(numeric_cols)} numeric columns...")

    # 3. Loop over each column to fill
    for col in numeric_cols:
        try:
            # Create a temporary copy for this column
            tmp = df[[col, 'minute_of_day', 'date']].copy()
            
            # Remove potential duplicate columns (robustness)
            tmp = tmp.loc[:, ~tmp.columns.duplicated()] 
            tmp['minute_of_day'] = tmp['minute_of_day'].astype(int)

            # 4. Create the Pivot Grid (Days x Minutes)
            pivot = tmp.pivot_table(
                index='date',
                columns='minute_of_day',
                values=col,
                aggfunc='mean'
            )

            # 5. Smart Fill: Calculate rolling average (e.g., ±3 days)
            # This fills NaNs with the average of the same minute on nearby days
            window_size = days_window * 2 + 1
            filled = pivot.rolling(window=window_size, min_periods=1, center=True).mean()
            
            # Use the filled values ONLY for missing (NaN) spots
            pivot = pivot.combine_first(filled)

            # 6. Un-pivot (Melt) the grid back to a long format
            melted = pivot.reset_index().melt(
                id_vars='date',
                var_name='minute_of_day',
                value_name=col
            )
            
            melted['minute_of_day'] = melted['minute_of_day'].astype(int)
            melted.set_index(['date', 'minute_of_day'], inplace=True)
            filled_cols.append(melted[col])

        except Exception as e:
            print(f"  > Warning: Skipped column {col}: {e}")
            continue

    # 7. Re-assemble the final DataFrame
    if not filled_cols:
        print("  > Warning: No numeric columns were processed.")
        # Return the time-indexed, sorted df
        return df.reset_index() 

    merged = pd.concat(filled_cols, axis=1).reset_index()
    
    # Re-create the 'time' index
    merged['time'] = pd.to_datetime(merged['date']) + pd.to_timedelta(merged['minute_of_day'], unit='m')
    merged = merged.drop(columns=['date', 'minute_of_day']).set_index('time').sort_index()
    
    # 8. Final safety fill for any remaining edge-case NaNs
    merged = merged.fillna(method='ffill').fillna(method='bfill')

    return merged.reset_index()



def fill_missing_values_for_all(df_list: List[pd.DataFrame], days_window: int = 7) -> List[pd.DataFrame]:
    """
    Processes a list of DataFrames, filling missing values in each 
    by calling the _safe_fill_single_file helper function.

    Args:
        df_list (List[pd.DataFrame]): The list of DataFrames to process.
        days_window (int): The window size (e.g., 3 = ±3 days). 
                           Defaults to 3.

    Returns:
        List[pd.DataFrame]: A new list containing the filled DataFrames.
    """
    
    filled_dfs_list = []
    total_files = len(df_list)
    
    print(f"---  Starting Missing Value Fill Process for {total_files} files ---")
    print(f"--- Window size: ±{days_window} days ---")

    # This loop was in your main script, now it's inside the function
    for i, df in enumerate(df_list, start=1):
        print(f"\nProcessing file {i}/{total_files} (Original Shape: {df.shape})...")
        try:
            # Call the *internal* helper function
            filled_df = _safe_fill_single_file(df, days_window=days_window)
            
            filled_dfs_list.append(filled_df)
            print(f" File {i} done — New shape: {filled_df.shape}")
        
        except Exception as e:
            # Catch errors for a specific file and continue
            print(f" ERROR: Skipped file {i} due to an unexpected error: {e}")
            continue
    
    print("\n---  All files processed. ---")
    return filled_dfs_list