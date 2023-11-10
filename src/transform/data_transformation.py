import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin

class RULAdder(BaseEstimator, TransformerMixin):
    """
    Add the RUL column to the DataFrame.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Calculate max_cycles
        max_cycles = X.groupby('Engine')['Cycle'].max()

        # Merge with the original DataFrame
        X = X.merge(max_cycles.to_frame(name='max_cycle'), left_on='Engine', right_index=True)

        # Calculate RUL
        X['RUL'] = X['max_cycle'] - X['Cycle']

        # Drop the max_cycle column and return the resulting DataFrame
        return X.drop('max_cycle', axis=1)


class ConstantColumnDropper(BaseEstimator, TransformerMixin):
    """
    Transformer to drop columns that show a constant value.
    This suggests there is no change as the RUL reaches failure.
    """

    def fit(self, X, y=None):
        # Determine columns with constant values
        self.cols_to_drop_ = X.columns[X.nunique() <= 1].tolist()

        # Always drop the column named '(Physical Core Speed) (rpm)' if it exists
        if '(Physical Core Speed) (rpm)' in X.columns:
            self.cols_to_drop_.append('(Physical Core Speed) (rpm)')

        return self

    def transform(self, X):
        # Drop the columns identified in the fit method
        return X.drop(columns=self.cols_to_drop_)


class SequenceCreator(BaseEstimator, TransformerMixin):
    """
    Transforms a DataFrame of engine data into sequences of a given length.
    Each sequence does not include the 'RUL' column.
    """
    def __init__(self, sequence_length=50):
        self.sequence_length = sequence_length

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        """
        This function will only return sequences.
        Useful when running predictions or generating sequences.
        """
        sequences = []

        for engine in df['Engine'].unique():
            engine_data = df[df['Engine'] == engine].reset_index(drop=True)

            for i in range(len(engine_data) - self.sequence_length + 1):
                sequence = engine_data.iloc[i:i+self.sequence_length].drop(['RUL'], axis=1)

                sequences.append(sequence)

        return np.array(sequences)


    def transform_with_labels(self, df):
        """
        Use this function outside the pipeline context.
        This is when you need both the sequences and labels (like during training).
        """
        sequences = []
        labels = []

        for engine in df['Engine'].unique():
            engine_data = df[df['Engine'] == engine].reset_index(drop=True)

            for i in range(len(engine_data) - self.sequence_length + 1):
                sequence = engine_data.iloc[i:i+self.sequence_length].drop(['RUL'], axis=1)
                label = engine_data.loc[i+self.sequence_length-1, 'RUL']

                sequences.append(sequence)
                labels.append(label)

        return np.array(sequences), np.array(labels)


class DataScaler(BaseEstimator, TransformerMixin):
    """
    Scales the input data using the MinMaxScaler. Designed to scale 3D data,
    where the last dimension is considered as features for scaling purposes.
    """
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit(self, X, y=None):
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.scaler.fit(X_reshaped)
        return self

    def transform(self, X):
        X_reshaped = X.reshape(-1, X.shape[-1])
        X_reshaped_scaled = self.scaler.transform(X_reshaped)
        return X_reshaped_scaled.reshape(X.shape)
