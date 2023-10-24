

######## Helper Functions File  #################################

import os            
import psutil        
import numpy as np   
import pandas as pd

def get_memory_usage():
    """
    Get Memory Usage
    
    This function returns the memory usage of the current process in gigabytes (GB).

    Returns:
        float: The memory usage in gigabytes (GB).
    """
    # Retrieve memory information of the current process (in bytes)
    memory_info = psutil.Process(os.getpid()).memory_info()
    
    # Convert memory usage to gigabytes (GB) and round it to 2 decimal places
    memory_usage_gb = np.round(memory_info[0] / 2.**30, 2)
    
    return memory_usage_gb



def size_of_fmt(num, suffix='B'):
    """
    Format Memory Size
    
    This function formats a memory size (in bytes) into a human-readable string.
    
    Args:
        num (float): The memory size in bytes.
        suffix (str): The unit suffix for the formatted string (default is 'B' for bytes).

    Returns:
        str: A human-readable string representing the memory size.
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)



def reduce_dtypes(df, verbose=True):
    """
    Reduce Memory Usage of a Pandas DataFrame
    
    This function takes a pandas DataFrame and reduces its memory usage by changing data types.

    Args:
        df (pd.DataFrame): The input DataFrame to reduce memory usage.
        verbose (bool): If True, print memory reduction information (default is True).

    Returns:
        pd.DataFrame: The DataFrame with reduced memory usage.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    # Calculate the initial memory usage of the DataFrame
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Check if the column can be converted to a smaller integer type
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                # Check if the column can be converted to a smaller float type
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
   
    # Calculate the final memory usage of the DataFrame after reduction
    end_mem = df.memory_usage().sum() / 1024**2
    
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df





def merge_by_concat(df1, df2, merge_on):
    """
    Merge DataFrames by Concatenation to Preserve Data Types
    
    This function merges two DataFrames while preserving the data types of the columns.
    
    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame to be merged.
        merge_on (str or list of str): The column(s) on which to perform the merge.

    Returns:
        pd.DataFrame: The merged DataFrame with preserved data types.
    """
    # Create a temporary DataFrame with only the columns to be merged
    merged_gf = df1[merge_on]
    
    # Merge the temporary DataFrame with the second DataFrame
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    
    # Identify new columns in the merged DataFrame
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    
    # Concatenate the new columns from the merged DataFrame to the original DataFrame
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    
    return df1

