# Jet Engine Degradation Prediction

![](docs/renato-boemer-jet-engine-data-rnn.jpeg)

Prognostics and health management is an important topic in industry for predicting state of assets to avoid downtime and failures. This data set is the Kaggle version of the very well known public data set for asset degradation modeling from NASA. It includes Run-to-Failure simulated data from turbo fan jet engines.
Engine degradation simulation was carried out using C-MAPSS. Four different were sets simulated under different combinations of operational conditions and fault modes. Records several sensor channels to characterize fault evolution. The data set was provided by the Prognostics CoE at NASA Ames.

## Prediction Goal
The goal is to predict the remaining useful life (RUL) of turbo fan jet engines using NASA's C-MAPSS simulated sensor data. The RUL is equivalent to the number of flights remained for an engine after the last datapoint in the test dataset.

## Installation
You can clone this repository using git:
```$ git clone https://github.com/boemer00/jet-engine-degradation-prediction.git```

Then, donwload the dataset directly from [NASA's repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

## API Integration
The project now includes a FastAPI integration, allowing for model predictions through a REST API.

### Running the API Server

- Navigate to the project root directory.
- Start the FastAPI server by running:
`uvicorn src.app.app:app --reload`

This will host the API at `http://127.0.0.1:8000`.

### Interacting with the API
Access the auto-generated Swagger UI at `http://127.0.0.1:8000/docs` for an interactive interface to test API endpoints.

Use tools like curl or Postman to send POST requests to `http://127.0.0.1:8000/predict` with JSON-formatted input data.

## Docker Usage
To simplify the setup and execution process, Docker can be used:

### Build the Docker Image
Navigate to the directory containing the Dockerfile and execute:

`docker build -t jet-engine-prediction .`

### Run the Application in a Docker Container
Execute the application with:

`docker run -it --rm --name jet-engine-prediction-instance jet-engine-prediction`

# Model Architecture

This model is a Transformer-based neural network implemented using Keras. The architecture is described as follows:

### Transformer Block

- Type: Transformer block consisting of multi-head self-attention and position-wise feed-forward networks.

- Multi-Head Attention:
  - Number of Heads: 4 (configurable).
  - Key Dimension: 64 (configurable).
  - Dropout: 0.1 (configurable), applied to the attention weights.

- Feed-Forward Network:
  - Conv1D with 19 filters, kernel size of 1, and ReLU activation function.
  - Dropout: Applied post-activation with a rate of 0.1 (configurable).

- Residual Connections: Each sub-layer (i.e., multi-head attention, feed-forward) has a residual connection followed by layer normalization.

- Number of Transformer Blocks: 4 (configurable).

### Global Average Pooling Layer

- Computes the average of the features dimension assuming `channels_first` data format, resulting in a fixed-length output vector which helps to mitigate overfitting and reduces the total number of parameters.

### Dense Layers

- After the Global Average Pooling, the model includes dense layers with the following units: `[128]` (configurable). Each dense layer is followed by a ReLU activation and dropout for regularization.

### Output Layer

- Units: A single unit with no activation function (linear), suitable for regression tasks like predicting the Remaining Useful Life (RUL) in a jet engine.

## Compilation Details

- Loss Function: Mean Squared Error (MSE), which is suitable for regression tasks.
- Optimizer: Adam optimizer, which is computationally efficient and has little memory requirement. The learning rate is configurable.
- Metrics: The model evaluates Mean Absolute Error (MAE) during training, which provides a straightforward way to measure the average magnitude of errors in a set of predictions, without considering their direction.

Please note that the hyperparameters such as the number of heads, the dimension of the key, the number of transformer blocks, and the dropout rates are all configurable and can be tuned according to the specific requirements of the task at hand.


## RNN Pipeline Parameterization
You can customise the training of the RNN model through command-line parameters when running the pipeline script. While the script provides a set of default configurations, you can override them using arguments.

**Available Command Line Parameters:**
The parameters you can adjust via command line include, but are not limited to:
- ```--learning_rate```
- ```--epochs```
- ```--batch_size```

**Usage Example:**
To override the default learning rate, run:
```python pipelines/rnn_pipeline.py --learning_rate 0.1``

**Default Configurations:**
For a comprehensive list of default configurations and their values, please refer to the ```config.yaml``` file in the project directory.


## Running Tests
To run tests for this project, ensure you have Python installed and the required packages from *requirements.txt*. Navigate to the project directory and activate a virtual environment (recommended). Use the pytest command to execute tests. For detailed output, use ```pytest -v```. If using pytest-cov for test coverage, view the report with ```pytest --cov=src```

## Experiment Tracking with MLflow

This project utilises MLflow for experiment tracking within the `pipeline.py` script. To run experiments and view results:

1. **Set Up MLflow**:
   - Install MLflow with `pip install mlflow`.

2. **Run the MLflow-Enabled Script**:
   - Execute the `pipeline.py` script to kick off experiments:
     ```bash python pipeline.py```

3. **Start MLflow Server**:
   - To visualise the experiment results, initiate the MLflow tracking server:
     ```bashmlflow ui```

4. **Access UI**:
   - Visit `http://127.0.0.1:5000` to view the MLflow UI and inspect your experiment runs.

## Results
Upon evaluating the Remaining Useful Life (RUL) prediction model against the training dataset, several key observations have been made. The training data exhibits a mean RUL of 107.81, a median of 103.00, and spans a range from 0 to 361. The model, when tested, yielded a **Mean Absolute Error (MAE) of 5.8521** versus the LSTM's baseline of 12.8381.

Given that the mean RUL is approximately 108, an MAE of around 6 (to the nearest integer) represents a typical error of about 6% of the mean RUL. In comparison to the broad range of RUL values (361), this MAE indicates a relatively good prediction error. Nevertheless, further iterations and refinements can also be pursued to improve the model's performance.

## References
A. Saxena, K. Goebel, D. Simon, and N. Eklund, *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.
