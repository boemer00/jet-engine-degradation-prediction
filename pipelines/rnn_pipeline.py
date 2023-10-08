import mlflow
from sklearn.pipeline import Pipeline
from src.data.data_loading import load_data
from src.transform.data_transformation import RULAdder, ConstantColumnDropper, SequenceCreator, DataSplitter, DataScaler
from src.models.rnn_model import initialize_model, train_model

# # Set MLflow Tracking URI and Experiment
# mlflow.set_tracking_uri('file:///path/to/directory')
# mlflow.set_experiment('rb-jet-engine')

def main():
    # 1. Data Loading
    raw_data = load_data()

    # 2. Data Pre-processing
    transformer_pipeline = Pipeline([
        ('add_rul', RULAdder()),
        ('drop_constants', ConstantColumnDropper()),
        ('create_sequences', SequenceCreator(sequence_length=50)),
        ('split_data', DataSplitter(test_size=0.2, random_state=42)),
        ('scale_data', DataScaler())
    ])
    X_train_scaled, X_test_scaled, y_train, y_test = transformer_pipeline.transform(raw_data)

    # 3. Model Initialisation
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    model = initialize_model(input_shape)

    # 4. Model Training
    trained_model = train_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Log model on MLFlow? (TBC)

if __name__ == "__main__":
    main()
