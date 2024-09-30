import pandas as pd 
from pandas import DataFrame

def get_dataframe_metadata(df: DataFrame):
    # Check if the DataFrame has a 'name' attribute
    table_name = getattr(df, 'name', 'Unnamed')
    
    metadata = f"The table's name is {table_name}.\n"
    metadata += f"Total number of rows: {len(df)}\n"
    metadata += "List of columns:\n"
    
    # Iterate through each column and gather information
    for column in df.columns:
        # Get column data type
        dtype = df[column].dtype
        
        # Number of non-null entries
        non_null_count = df[column].notnull().sum()
        
        # Minimum and maximum values, where applicable
        try:
            min_value = df[column].min()
            max_value = df[column].max()
        except TypeError:
            min_value = 'N/A'
            max_value = 'N/A'
        
        # Sample data (up to three entries)
        sample_data = df[column].dropna().head(3).tolist()
        sample_data_str = ", ".join(map(str, sample_data))
        
        # Compile column metadata
        metadata += f"Column '{column}': Datatype={dtype}, Non-null Count={non_null_count}, Min={min_value}, Max={max_value}, Sample Data=[{sample_data_str}]\n"
    
    return metadata