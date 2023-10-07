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
    # RNN Architecture
    model = Sequential()

    # LSTM layer with L2 regularization
    model.add(layers.LSTM(units=30,
                          activation='tanh',
                          kernel_regularizer=regularizers.l2(0.01),
                          input_shape=input_shape))

    # Dropout layer to reduce overfitting
    model.add(layers.Dropout(0.2))

    # Batch normalisation layer
    model.add(layers.BatchNormalization())

    # Dense output layer
    model.add(layers.Dense(1, activation='linear'))

    # Compile
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
    - history (History): Training history.
    """
    history = model.fit(X_train_scaled, y_train,
                        validation_data=(X_test_scaled, y_test),
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=0)

    return history
