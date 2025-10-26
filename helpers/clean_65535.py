import numpy as np
import pandas as pd

def replace_value_in_dfs(df_list, value_to_replace, replacement_value):
    """
    Loops through a list of DataFrames and replaces a specific value
    with a new value (inplace).

    Args:
        df_list (list): A list containing the pandas DataFrames to clean.
        value_to_replace: The value to search for (e.g., 65535).
        replacement_value: The new value to replace with (e.g., np.nan).
    """
    
    print(f"--- Cleaning Started: Replacing {value_to_replace} with {replacement_value} ---")

    # --- Loop through each dataframe and perform the replacement ---
    # We use enumerate to get an index (i) for printing
    for i, df in enumerate(df_list):
        
        if not isinstance(df, pd.DataFrame):
            print(f"  Warning: Item #{i+1} is not a DataFrame. Skipping.")
            continue
            
        # inplace=True modifies the original DataFrame directly
        df.replace(value_to_replace, replacement_value, inplace=True)
        
        print(f"  Processed DataFrame #{i+1}: All {value_to_replace} values replaced.")

    print("\n--- Cleaning Complete ---")