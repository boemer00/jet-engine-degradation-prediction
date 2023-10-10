import pandas as pd
import numpy as np
from src.transform.data_transformation import RULAdder, ConstantColumnDropper
from src.transform.data_transformation import SequenceCreator, DataScaler

def test_RULAdder():
    transformer = RULAdder()

    # Create a sample DataFrame
    df = pd.DataFrame({
        'Engine': [1, 1, 1, 2, 2, 2],
        'Cycle': [1, 2, 3, 1, 2, 3]
    })

    # Apply RULAdder
    transformed_df = transformer.transform(df)

    # Assertions
    assert 'RUL' in transformed_df.columns
    assert 'max_cycle' not in transformed_df.columns
    assert all(transformed_df.query("Engine == 1")['RUL'] == [2, 1, 0])
    assert all(transformed_df.query("Engine == 2")['RUL'] == [2, 1, 0])


def test_ConstantColumnDropper():
    transformer = ConstantColumnDropper()

    # Create a sample DataFrame
    df = pd.DataFrame({
        'Engine': [1, 1, 1],
        'Cycle': [1, 2, 3],
        'Constant': [5, 5, 5],
        '(Physical Core Speed) (rpm)': [1000, 1000, 1000]
    })

    # Apply ConstantColumnDropper
    transformed_df = transformer.fit_transform(df)

    # Assertions
    assert 'Constant' not in transformed_df.columns
    assert '(Physical Core Speed) (rpm)' not in transformed_df.columns


def test_SequenceCreator():
    transformer = SequenceCreator(sequence_length=2)

    # Create a sample DataFrame
    df = pd.DataFrame({
        'Engine': [1, 1, 2, 2],
        'Cycle': [1, 2, 1, 2],
        'RUL': [1, 0, 1, 0]
    })

    # Apply SequenceCreator
    sequences, labels = transformer.transform_with_labels(df)

    # Assertions for transform_with_labels method
    assert sequences.shape == (2, 2, 2)
    assert all(labels == [0, 0])


def test_DataScaler():
    transformer = DataScaler()

    # Create a sample 3D array
    X = np.array([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])

    # Apply DataScaler
    transformed_X = transformer.fit_transform(X)

    # Assertions
    assert transformed_X.shape == X.shape
    assert transformed_X.min() >= 0
    assert transformed_X.max() <= 1
