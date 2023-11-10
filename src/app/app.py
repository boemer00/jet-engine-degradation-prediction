import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from src.transform.data_transformation import RULAdder, ConstantColumnDropper, SequenceCreator, DataScaler

# Define a Pydantic model for the input data
class Features(BaseModel):
    features: list

# Initialise the FastAPI application
app = FastAPI()

# Dynamically construct the absolute path to the model file
current_directory = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_directory, '..', '..'))
model_path = os.path.join(project_root, 'model.keras')

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Initialise the transformation objects
rul_adder = RULAdder()
constant_column_dropper = ConstantColumnDropper()
sequence_length = 50
sequence_creator = SequenceCreator(sequence_length=sequence_length)
data_scaler = DataScaler()

@app.post('/predict')
def predict(features: Features):
    try:
        # Convert features to numpy array
        feature_array = np.array(features.features)

        # 1. Add RUL
        feature_array_with_rul = rul_adder.transform(feature_array)

        # 2. Drop constant columns
        transformed_data = constant_column_dropper.transform(feature_array_with_rul)

        # 3. Create sequences
        sequences, _ = sequence_creator.transform_with_labels(transformed_data)

        # 4. Scale data
        sequences_scaled = data_scaler.transform(sequences)

        # Generate predictions
        predictions = model.predict(sequences_scaled)

        # Convert predictions to a list for JSON response
        predictions_list = predictions.tolist()

        # Send back the predictions as a response
        return {"predictions": predictions_list}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)



# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import tensorflow as tf
# import numpy as np

# # Define a Pydantic model for the input data
# class Features(BaseModel):
#     features: list

# # Initialise the FastAPI application
# app = FastAPI()

# # Load the trained model
# model = tf.keras.models.load_model('/jet-engine/model.keras')

# @app.post('/predict')
# def predict(features: Features):
#     try:
#         # Convert features to numpy array
#         feature_array = np.array(features.features)

#         # Perform any necessary reshaping, preprocessing here
#         # For example, if your model expects data of a certain shape
#         # feature_array = feature_array.reshape(expected_shape)

#         # Generate predictions
#         predictions = model.predict(feature_array)

#         # Convert predictions to a list for JSON response
#         predictions_list = predictions.tolist()

#         # Send back the predictions as a response
#         return {"predictions": predictions_list}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == '__main__':
#     import uvicorn
#     uvicorn.run(app, host='0.0.0.0', port=5000)
