# Jet Engine Degradation Prediction

![](docs/renato-boemer-jet-engine-data-rnn.jpeg)

Prognostics and health management is an important topic in industry for predicting state of assets to avoid downtime and failures. This data set is the Kaggle version of the very well known public data set for asset degradation modeling from NASA. It includes Run-to-Failure simulated data from turbo fan jet engines.
Engine degradation simulation was carried out using C-MAPSS. Four different were sets simulated under different combinations of operational conditions and fault modes. Records several sensor channels to characterize fault evolution. The data set was provided by the Prognostics CoE at NASA Ames.

## Prediction Goal
The goal is to predict the remaining useful life (RUL) of turbo fan jet engines using NASA's C-MAPSS simulated sensor data.
RUL is equivalent of number of flights remained for the engine after the last datapoint in the test dataset.

## Installation
You can clone this repository using git:
```$ git clone https://github.com/boemer00/jet-engine-degradation-prediction.git```

Then, donwload the dataset directly from [NASA's repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

## Model Architecture
This model is a simple recurrent neural network (RNN) constructed using Keras. The architecture can be described as follows:

**LSTM Layer**

- Type: Long Short-Term Memory (LSTM) layer.
- Units: 30 LSTM units.
- Activation: Hyperbolic Tangent (tanh) activation function.
- Regularization: L2 regularization with a coefficient of 0.01 applied to the kernel.
- Input Shape: Variable, as per input_shape (which represents (timesteps, features)).

**Dropout Layer**

- Dropout Rate: 20% dropout rate, which means during training, 20% of the units in the previous layer are randomly set to 0 at each update cycle. This is a regularization technique to prevent overfitting.

**Batch Normalization Layer**

This layer normalizes the activations of the previous layer at each batch, which can help in speeding up the training process and stabilizing the training of deep networks.

**Dense Layer**

- Units: A single unit.
- Activation: Linear activation function, which means the output is the raw weighted sum of the inputs.

**Compilation Details**

- Loss Function: Mean Squared Error (MSE) which is suitable for regression tasks.
- Optimizer: RMSprop optimizer. RMSprop adjusts the Adagrad method in a very simple way to reduce its aggressive, monotonically decreasing learning rate.
- Metrics: The model tracks Mean Absolute Error (MAE) as a metric, which provides a straightforward way to observe the average error in predictions.

## Running Tests
To run tests for this project, ensure you have Python installed and the required packages from ```requirements.txt```. Navigate to the project directory and activate a virtual environment (recommended). Use the pytest command to execute tests. For detailed output, use ```pytest -v```. If using pytest-cov for test coverage, view the report with ```pytest --cov=src```

## Results
Upon evaluating the Remaining Useful Life (RUL) prediction model against the training dataset, several key observations have been made. The training data exhibits a mean RUL of 107.81, a median of 103.00, and spans a range from 0 to 361. The model, when tested, yielded a **Root Mean Squared Error (RMSE) of 14.1751** using random_state=42.

Given that the mean RUL is approximately 108, an RMSE of around 14 represents a typical error of about 14% of the mean RUL. In comparison to the broad range of RUL values (361), this RMSE indicates a relatively modest prediction error. Further iterations and refinements can also be pursued to improve the model's performance.

## References
A. Saxena, K. Goebel, D. Simon, and N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
