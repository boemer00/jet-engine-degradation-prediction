from keras.models import Sequential
from keras import layers, regularizers

def initialize_model(input_shape):
    """
    Initialises the RNN model with the given input shape.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).

    Returns:
    - model (keras.models.Sequential): The initialised model.
    """
    model = Sequential()
    model.add(layers.LSTM(units=30,
                          activation='tanh',
                          kernel_regularizer=regularizers.l2(0.01),
                          input_shape=input_shape))

    model.add(layers.Dropout(0.2))

    model.add(layers.BatchNormalization())

    model.add(layers.Dense(1, activation='linear'))

    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=['mean_absolute_error'])

    return model

def train_model(model, X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, batch_size=160):
    """
    Trains the provided model with the given training and validation data.

    Parameters:
    - model (keras.models.Sequential): The model to train.
    - X_train_scaled (numpy.ndarray): Scaled training data.
    - y_train (numpy.ndarray): Training labels.
    - X_test_scaled (numpy.ndarray): Scaled validation data.
    - y_test (numpy.ndarray): Validation labels.
    - epochs (int, optional): Number of epochs to train. Default is 100.
    - batch_size (int, optional): Batch size for training. Default is 160.

    Returns:
    - Training model.
    """
    model.fit(X_train_scaled, y_train,
              validation_data=(X_test_scaled, y_test),
              epochs=epochs,
              batch_size=batch_size,
              verbose=0)

    return model

# import mlflow
# import mlflow.keras

# def train_model(model, X_train_scaled, y_train, X_test_scaled, y_test, epochs=100, batch_size=160):
#     """
#     Trains the provided model with the given training and validation data.
#     ... [rest of your docstring]
#     """

#     # Start a new run
#     with mlflow.start_run():
#         history = model.fit(X_train_scaled, y_train,
#                             validation_data=(X_test_scaled, y_test),
#                             epochs=epochs,
#                             batch_size=batch_size,
#                             verbose=0)

#         # Log parameters
#         mlflow.log_param("epochs", epochs)
#         mlflow.log_param("batch_size", batch_size)

#         # Log metrics
#         for epoch, loss in enumerate(history.history['loss']):
#             mlflow.log_metric("loss", loss, step=epoch)
#             mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

#         # Log model
#         mlflow.keras.log_model(model, "model")

#         # Optionally, you can also save and log other artifacts like plots or feature importance

#         # End the run
#         mlflow.end_run()

#     return model
