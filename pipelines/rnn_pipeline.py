import os
import yaml
import argparse
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from src.data.data_loading import load_train_data
from src.transform.data_transformation import RULAdder, ConstantColumnDropper, SequenceCreator, DataScaler
from src.models.rnn_model import initialize_model, train_model

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

# Set MLflow tracking directory
os.environ['MLFLOW_DIR'] = '/Users/renatoboemer/code/lewagon/jet-engine/mlruns'
MLFLOW_DIR = os.environ.get('MLFLOW_DIR', './mlruns')
mlflow.set_tracking_uri(f'file://{MLFLOW_DIR}')

def evaluate_model_mae(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f'Test MAE: {mae:.4f}')
    return mae

def main():
    mlflow.set_experiment('default_experiment')

    with mlflow.start_run():
        # 1. Data Loading
        raw_data = load_train_data()

        # 2. Data Transformation
        rul_adder = RULAdder()
        raw_data_with_rul = rul_adder.transform(raw_data)

        constant_column_dropper = ConstantColumnDropper()
        transformed_data = constant_column_dropper.fit_transform(raw_data_with_rul)

        sequence_creator = SequenceCreator(sequence_length=sequence_length)
        sequences, labels = sequence_creator.transform_with_labels(transformed_data)

        mlflow.log_param('sequence_length', sequence_length)

        # 3. Data Splitting and Scaling
        X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=test_size, random_state=random_state)

        mlflow.log_param('test_size', test_size)
        mlflow.log_param('random_state', random_state)

        data_scaler = DataScaler()
        X_train_scaled = data_scaler.fit_transform(X_train)
        X_test_scaled = data_scaler.transform(X_test)

        # 4. Model Initialisation
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])

        head_size = config['model']['head_size']
        num_heads = config['model']['num_heads']
        ff_dim = config['model']['ff_dim']
        num_transformer_blocks = config['model']['num_transformer_blocks']
        mlp_units = config['model']['mlp_units']
        dropout = config['model']['dropout']
        mlp_dropout = config['model']['mlp_dropout']

        model = initialize_model(input_shape, head_size, num_heads, ff_dim,
                                 num_transformer_blocks, mlp_units, dropout, mlp_dropout)

        # model = initialize_model(input_shape) lstm version

        # 5. Model Training
        trained_model = train_model(model, X_train_scaled, y_train, X_test_scaled, y_test, learning_rate, epochs, batch_size)

        # 6. Model Evaluation for MAE
        mae = evaluate_model_mae(trained_model, X_test_scaled, y_test)
        mlflow.log_metric('mae', mae)

        # 7. Save the trained model with timestamped filename
        save_dir = './models/saved'
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = f'{save_dir}/model_{timestamp}.keras'
        trained_model.save(save_path)

        # Optionally log the model with MLflow
        mlflow.keras.log_model(trained_model, 'model')

if __name__ == "__main__":
    main()
