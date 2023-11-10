from joblib import dump
from src.data.data_loading import load_train_data
from src.transform.data_transformation import RULAdder, ConstantColumnDropper, SequenceCreator, DataScaler

# Load training data
df_train = load_train_data()

# Initialize transformers
rul_adder = RULAdder()
constant_column_dropper = ConstantColumnDropper()
sequence_creator = SequenceCreator(sequence_length=50)
data_scaler = DataScaler()

# Apply transformations
transformed_data = rul_adder.transform(df_train)
transformed_data = constant_column_dropper.fit_transform(transformed_data)
sequences, _ = sequence_creator.transform_with_labels(transformed_data)

# Fit the scaler
data_scaler.fit(sequences)

# Save the fitted scaler
dump(data_scaler.scaler, 'scaler.joblib')
