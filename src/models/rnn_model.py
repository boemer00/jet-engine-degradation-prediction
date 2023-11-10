from keras import layers
import tensorflow as tf

def initialize_model(input_shape, head_size=64, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], dropout=0.1, mlp_dropout=0.1):
    """
    Initializes the Transformer model.

    Parameters:
    - input_shape (tuple): Shape of the input data (timesteps, features).
    - head_size (int): Dimensionality of the query-key-value.
    - num_heads (int): Number of attention heads.
    - ff_dim (int): Hidden layer size in feed forward network inside transformer.
    - num_transformer_blocks (int): Number of transformer blocks.
    - mlp_units (list): Number of dense units in MLP layers.
    - dropout (float): Dropout rate.
    - mlp_dropout (float): Dropout rate for MLP layers.

    Returns:
    - model (tf.keras.Model): The initialised model.
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=head_size, dropout=dropout
        )(x1, x1)

        # Skip connection 1.
        x2 = layers.Add()([attention_output, x])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP with the corrected number of filters.
        x3 = layers.Conv1D(filters=19, kernel_size=1, activation='relu')(x3)

        # Dropout.
        x3 = layers.Dropout(mlp_dropout)(x3)

        # Skip connection 2 with the correct shape.
        x = layers.Add()([x3, x2])

    # Create a [batch_size, features] tensor.
    x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
    for units in mlp_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def train_model(model, X_train, y_train, X_test, y_test, learning_rate, epochs=100, batch_size=32):
    """
    Trains the provided model with the given training and validation data.

    Parameters:
    - model (tf.keras.Model): The model to train.
    - X_train (numpy.ndarray): Training data.
    - y_train (numpy.ndarray): Training labels.
    - X_test (numpy.ndarray): Validation data.
    - y_test (numpy.ndarray): Validation labels.
    - learning_rate (float): Learning rate for the optimizer.
    - epochs (int, optional): Number of epochs to train. Default is 100.
    - batch_size (int, optional): Batch size for training. Default is 32.

    Returns:
    - Trained model.
    """
    # Compile the model.
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error'])

    # Train the model.
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1)

    return model

## ------------------------------------------ ##

# import tensorflow as tf
# from keras.models import Sequential
# from keras import layers, regularizers

# def initialize_model(input_shape):
#     """
#     Initialises the RNN model with the given input shape.

#     Parameters:
#     - input_shape (tuple): Shape of the input data (timesteps, features).

#     Returns:
#     - model (keras.models.Sequential): The initialised model.
#     """
#     model = Sequential()
#     model.add(layers.LSTM(units=30,
#                           activation='tanh',
#                           kernel_regularizer=regularizers.l2(0.01),
#                           input_shape=input_shape))
#     model.add(layers.Dropout(0.2))
#     model.add(layers.BatchNormalization())
#     model.add(layers.Dense(1, activation='linear'))

#     return model

# def train_model(model, X_train_scaled, y_train, X_test_scaled, y_test, learning_rate, epochs=100, batch_size=160):
#     """
#     Trains the provided model with the given training and validation data.

#     Parameters:
#     - model (keras.models.Sequential): The model to train.
#     - X_train_scaled (numpy.ndarray): Scaled training data.
#     - y_train (numpy.ndarray): Training labels.
#     - X_test_scaled (numpy.ndarray): Scaled validation data.
#     - y_test (numpy.ndarray): Validation labels.
#     - learning_rate (float): Learning rate for the optimizer.
#     - epochs (int, optional): Number of epochs to train. Default is 100.
#     - batch_size (int, optional): Batch size for training. Default is 160.

#     Returns:
#     - Trained model.
#     """

#     # Compile the model right before training.
#     optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
#     model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mean_absolute_error'])

#     model.fit(X_train_scaled, y_train,
#               validation_data=(X_test_scaled, y_test),
#               epochs=epochs,
#               batch_size=batch_size,
#               verbose=0)

#     return model
