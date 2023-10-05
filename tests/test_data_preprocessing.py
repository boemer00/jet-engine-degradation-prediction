import pandas as pd
from transform.data_transformation import add_RUL_column

def test_basic_RUL_calculation():
    df = pd.DataFrame({
        'Engine': [1, 1, 1],
        'Cycle': [1, 2, 3]
    })

    transformed_df = add_RUL_column(df)

    # Check if RUL is correctly computed
    assert all(transformed_df['RUL'] == [2, 1, 0])

def test_column_presence():
    df = pd.DataFrame({
        'Engine': [1, 1, 1],
        'Cycle': [1, 2, 3]
    })

    transformed_df = add_RUL_column(df)

    # Check presence of RUL and absence of max_cycle
    assert 'RUL' in transformed_df.columns
    assert 'max_cycle' not in transformed_df.columns

def test_engine_grouping():
    df = pd.DataFrame({
        'Engine': [1, 1, 2, 2, 2],
        'Cycle': [1, 2, 1, 2, 3]
    })

    transformed_df = add_RUL_column(df)

    expected_RULs = [1, 0, 2, 1, 0]
    assert all(transformed_df['RUL'] == expected_RULs)

def test_empty_dataframe():
    df = pd.DataFrame({
        'Engine': [],
        'Cycle': []
    })

    transformed_df = add_RUL_column(df)

    # Ensure dataframe is still empty and doesn't have RUL column
    assert transformed_df.empty
    assert 'RUL' not in transformed_df.columns
