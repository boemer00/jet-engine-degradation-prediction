import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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

def drop_constant_cols(df):
    """
    Drop columns that show a constant value.
    It suggests there is not change as the RUL reaches failure.
    """
    # Get columns with zero variance (constant value)
    cols_to_drop = df.columns[(df.nunique() <= 1) | (df.columns == '(Physical Core Speed) (rpm)')]

    # Drop the constant columns and return the resulting DataFrame
    return df.drop(columns=cols_to_drop)

from sklearn.model_selection import train_test_split

def create_sequences(df, sequence_length=50):
    """
    Create sequences of a given length from the DataFrame.

    Parameters:
    - df (DataFrame): Input DataFrame containing engine data.
    - sequence_length (int, optional): Length of sequences to be created. Default is 50.

    Returns:
    - sequences (numpy.ndarray): 3D array containing sequences.
    - labels (numpy.ndarray): 1D array containing corresponding labels for sequences.
    """

    sequences = []
    labels = []

    for engine in df['Engine'].unique():
        engine_data = df[df['Engine'] == engine].reset_index(drop=True)

        for i in range(len(engine_data) - sequence_length + 1):
            sequence = engine_data.iloc[i:i+sequence_length].drop(['RUL'], axis=1)
            label = engine_data.loc[i+sequence_length-1, 'RUL']

            sequences.append(sequence)
            labels.append(label)

    return np.array(sequences), np.array(labels)

def split_data(df, test_size=0.2, random_state=42, sequence_length=50):
    """
    Split the engine data into training and test sequences.

    Parameters:
    - df (DataFrame): Input DataFrame containing engine data.
    - test_size (float, optional): Proportion of the data to be used as the test set. Default is 0.2.
    - random_state (int, optional): Random seed for reproducibility. Default is 42.
    - sequence_length (int, optional): Length of sequences to be created. Default is 50.

    Returns:
    - X_train, X_test (numpy.ndarray): Training and test sequences.
    - y_train, y_test (numpy.ndarray): Corresponding labels for the training and test sequences.
    """

    sequences, labels = create_sequences(df, sequence_length)
    return train_test_split(sequences, labels, test_size=test_size, random_state=random_state)


def scale_data(X_train, X_test):
    scaler = MinMaxScaler()

    # Reshape, fit and transform on train data
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    X_train_reshaped_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_reshaped_scaled.reshape(X_train.shape)

    # Only transform on test data using the scaler fitted on train data
    X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
    X_test_reshaped_scaled = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_reshaped_scaled.reshape(X_test.shape)

    return X_train_scaled, X_test_scaled, scaler

# X_train_scaled, X_test_scaled, scaler_used = scale_data(X_train, X_test)
