import yaml
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from src.data.data_loading import load_train_data
from src.transform.data_transformation import RULAdder, ConstantColumnDropper, SequenceCreator, DataScaler

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Argument Parsing
parser = argparse.ArgumentParser(description='Train ML model with parameters from config file.')
parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
parser.add_argument('--learning_rate', type=float, help='Override learning rate')
parser.add_argument('--epochs', type=int, help='Override number of epochs')
parser.add_argument('--batch_size', type=int, help='Override batch size')
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Override parameters if provided in command-line
sequence_length = config['data']['sequence_length']
test_size = config['data']['test_size']
random_state = config['data']['random_state']
learning_rate = args.learning_rate or config['training']['learning_rate']
epochs = args.epochs or config['training']['epochs']
batch_size = args.batch_size or config['training']['batch_size']

def evaluate_model_mae(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f'Test MAE: {mae:.4f}')
    return mae

if __name__ == "__main__":

    # Data Transformation
    rul_adder = RULAdder()
    raw_data_with_rul = rul_adder.transform(load_train_data())

    constant_column_dropper = ConstantColumnDropper()
    transformed_data = constant_column_dropper.fit_transform(raw_data_with_rul)

    sequence_creator = SequenceCreator(sequence_length=sequence_length)
    sequences, labels = sequence_creator.transform_with_labels(transformed_data)

    # Data Splitting and Scaling
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=test_size, random_state=random_state)

    data_scaler = DataScaler()
    X_train_scaled = data_scaler.fit_transform(X_train)
    X_test_scaled = data_scaler.transform(X_test)

    # Load the trained model
    model = tf.keras.models.load_model('/Users/renatoboemer/code/lewagon/jet-engine/models/saved/model_20231108_190623.keras')

    # Evaluate the model for MAE
    mae = evaluate_model_mae(model, X_test_scaled, y_test)
    print(f'The final MAE is: {mae}')
