import pandas as pd
from typing import List

def get_statistical_summary(df_list: List[pd.DataFrame], name_list: List[str]) -> pd.DataFrame:
    """
    Generates a combined statistical summary (.describe()) for a list of DataFrames.

    It loops through each DataFrame, calculates its summary, adds a 'file_source'
    column based on the provided name_list, and then concatenates all
    summaries into one master DataFrame.

    Args:
        df_list (List[pd.DataFrame]): A list of the DataFrames to analyze.
        name_list (List[str]): A list of names corresponding to each DataFrame.

    Returns:
        pd.DataFrame: A single, combined DataFrame containing the 
                      statistical summaries of all input DataFrames.
    """
    
    if len(df_list) != len(name_list):
        print("Warning: The number of DataFrames does not match the number of names.")
        return pd.DataFrame() # Return empty DataFrame on error

    # This list will hold the summary statistics for each dataframe
    all_summaries = []

    print(f"--- Generating Statistical Summary for {len(df_list)} files ---")

    # Step 1: Loop through each dataframe to calculate and collect its statistics
    for df, name in zip(df_list, name_list):
        # .T transposes the summary to have columns as rows
        summary = df.describe().T 
        
        # Add the 'file_source' column for identification
        summary['file_source'] = name 
        
        all_summaries.append(summary)

    # Step 2: Combine all summaries into one master table
    comparison_summary = pd.concat(all_summaries)

    print("--- Summary Generation Complete ---")
    
    # Return the final table
    return comparison_summary