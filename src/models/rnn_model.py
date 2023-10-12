import tensorflow as tf
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

    return model

def train_model(model, X_train_scaled, y_train, X_test_scaled, y_test, learning_rate, epochs=100, batch_size=160):
    """
    Trains the provided model with the given training and validation data.

    Parameters:
    - model (keras.models.Sequential): The model to train.
    - X_train_scaled (numpy.ndarray): Scaled training data.
    - y_train (numpy.ndarray): Training labels.
    - X_test_scaled (numpy.ndarray): Scaled validation data.
    - y_test (numpy.ndarray): Validation labels.
    - learning_rate (float): Learning rate for the optimizer.
    - epochs (int, optional): Number of epochs to train. Default is 100.
    - batch_size (int, optional): Batch size for training. Default is 160.

    Returns:
    - Trained model.
    """

    # Compile the model right before training.
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])

    model.fit(X_train_scaled, y_train,
              validation_data=(X_test_scaled, y_test),
              epochs=epochs,
              batch_size=batch_size,
              verbose=0)

    return model
