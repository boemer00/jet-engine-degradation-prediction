import os

def drop_test(df):
    """
    Drop columns that have only NaN.
    New data could be different: check tests directory.
    """
    col_names = ['Sensor 26', 'Sensor 27']
    cols_to_drop = [col for col in col_names if df[col].isna().sum() == df.shape[0]]
    return df.drop(columns=cols_to_drop)

def add_RUL_column(df):
    """
    The RUL corresponds to the remaining cycles for each unit before the engine fails (target)

    Parameters:
    - df: The input dataframe containing engine data.

    Returns:
    - pd.DataFrame: Dataframe with an additional column for RUL.
    """

    # Calculate max_cycles
    max_cycles = df.groupby('Engine')['Cycle'].max()

    # Merge with the original DataFrame
    merged = df.merge(max_cycles.to_frame(name='max_cycle'), left_on='Engine', right_index=True)

    # Calculate RUL
    merged['RUL'] = merged['max_cycle'] - merged['Cycle']

    # Drop the max_cycle column
    merged = merged.drop('max_cycle', axis=1)

    return merged
