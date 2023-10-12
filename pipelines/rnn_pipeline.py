import os
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from src.data.data_loading import load_train_data
from src.transform.data_transformation import RULAdder, ConstantColumnDropper, SequenceCreator, DataScaler
from src.models.rnn_model import initialize_model, train_model

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set MLflow tracking directory
os.environ['MLFLOW_DIR'] = '/Users/renatoboemer/code/lewagon/jet-engine/mlruns'
MLFLOW_DIR = os.environ.get('MLFLOW_DIR', './mlruns')
mlflow.set_tracking_uri(f'file://{MLFLOW_DIR}')

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    print(f'Test RMSE: {rmse:.4f}')
    return rmse


# def main():
#     # 1. Data Loading
#     raw_data = load_train_data()

#     # 2. Data Transformation
#     rul_adder = RULAdder()
#     raw_data_with_rul = rul_adder.transform(raw_data)

#     constant_column_dropper = ConstantColumnDropper()
#     transformed_data = constant_column_dropper.fit_transform(raw_data_with_rul)

#     sequence_creator = SequenceCreator(sequence_length=50)
#     sequences, labels = sequence_creator.transform_with_labels(transformed_data)

#     # 3. Data Splitting and Scaling
#     X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

#     data_scaler = DataScaler()
#     X_train_scaled = data_scaler.fit_transform(X_train)
#     X_test_scaled = data_scaler.transform(X_test)

#     # 4. Model Initialisation
#     input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
#     model = initialize_model(input_shape)

#     # 5. Model Training
#     trained_model = train_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

#     # 6. Model Evaluation
#     evaluate_model(trained_model, X_test_scaled, y_test)

#     # 7. Save the trained model with timestamped filename
#     save_dir = './models/saved'
#     os.makedirs(save_dir, exist_ok=True)

#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     save_path = f'{save_dir}/model_{timestamp}.keras'
#     trained_model.save(save_path)

def main():
    mlflow.set_experiment('default_experiment')

    with mlflow.start_run():   # Starting a new MLflow run
        # 1. Data Loading
        raw_data = load_train_data()

        # 2. Data Transformation
        rul_adder = RULAdder()
        raw_data_with_rul = rul_adder.transform(raw_data)

        constant_column_dropper = ConstantColumnDropper()
        transformed_data = constant_column_dropper.fit_transform(raw_data_with_rul)

        sequence_creator = SequenceCreator(sequence_length=50)
        sequences, labels = sequence_creator.transform_with_labels(transformed_data)

        # Log parameter
        mlflow.log_param('sequence_length', 50)

        # 3. Data Splitting and Scaling
        X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)
        # Log split ratio and random seed
        mlflow.log_param('test_size', 0.2)
        mlflow.log_param('random_state', 42)

        data_scaler = DataScaler()
        X_train_scaled = data_scaler.fit_transform(X_train)
        X_test_scaled = data_scaler.transform(X_test)

        # 4. Model Initialisation
        input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
        model = initialize_model(input_shape)

        # 5. Model Training
        trained_model = train_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

        # 6. Model Evaluation
        rmse = evaluate_model(trained_model, X_test_scaled, y_test)
        mlflow.log_metric('rmse', rmse)

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
