import mlflow
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.data.data_loading import load_train_data
from src.transform.data_transformation import RULAdder, ConstantColumnDropper, SequenceCreator, DataSplitter, DataScaler
from src.models.rnn_model import initialize_model, train_model

# # Set MLflow Tracking URI and Experiment
# mlflow.set_tracking_uri('file:///path/to/directory')
# mlflow.set_experiment('rb-jet-engine')

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0) # verbose=0 to suppress the progress bar
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    return loss, accuracy

def main():
    # 1. Data Loading
    raw_data = load_train_data()

    # 2. Data Transformation
    rul_adder = RULAdder()
    raw_data_with_rul = rul_adder.transform(raw_data)

    constant_column_dropper = ConstantColumnDropper()
    transformed_data = constant_column_dropper.fit_transform(raw_data_with_rul)

    sequence_creator = SequenceCreator(sequence_length=50)
    sequences, labels = sequence_creator.transform_with_labels(transformed_data)

    # 3. Data Splitting and Scaling
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    data_scaler = DataScaler()
    X_train_scaled = data_scaler.fit_transform(X_train)
    X_test_scaled = data_scaler.transform(X_test)

    # 4. Model Initialisation
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    model = initialize_model(input_shape)

    # 5. Model Training
    trained_model = train_model(model, X_train_scaled, y_train, X_test_scaled, y_test)

    # 6. Model Evaluation
    evaluate_model(trained_model, X_test_scaled, y_test)

    # 7. Save the trained model (uncomment if needed)
    # save_model(trained_model, "path_to_save_model.h5")

if __name__ == "__main__":
    main()
