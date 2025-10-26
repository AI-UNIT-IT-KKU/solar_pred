import pandas as pd

def final_cleanse(df: pd.DataFrame) -> pd.DataFrame:
    """
    Execute a final cleansing on the DataFrame based on a dual strategy:
    1. Remove the specified catastrophic minute.
    2. Resolve remaining conflicts using the median.
    3. Merge the processed data with the original non-duplicate data.
    """
    
    # Determine the catastrophic time to be removed
    catastrophic_time = pd.to_datetime('2020-01-29 23:09:00')

    
    # 1. Separate the data into duplicate and non-duplicate sets
    non_duplicates = df[~df['time'].duplicated(keep=False)]
    duplicates = df[df['time'].duplicated(keep=False)]


    # 2. Process the duplicates
    duplicates_to_process = duplicates[duplicates['time'] != catastrophic_time]

    if duplicates_to_process.empty:
        print("No other duplicates to process after removing catastrophic time.")
        
        clean_non_duplicates = non_duplicates[non_duplicates['time'] != catastrophic_time]
        
        print("Returning only non-duplicate data (catastrophic time removed).")
        return clean_non_duplicates.reset_index(drop=True) 

    resolved_duplicates = duplicates_to_process.groupby('time').median()
    
    # 3. Merge the non-duplicate data with the resolved duplicate data
    clean_non_duplicates = non_duplicates[non_duplicates['time'] != catastrophic_time]
    
    final_df = pd.concat([clean_non_duplicates, resolved_duplicates.reset_index()], ignore_index=True)
    
    final_df = final_df.sort_values(by='time').reset_index(drop=True)
    
    return final_df