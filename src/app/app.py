import os
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
import tensorflow as tf
import numpy as np
from src.transform.data_transformation import RULAdder, ConstantColumnDropper, SequenceCreator, DataScaler

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
async def predict(file: UploadFile = File(...)):
    try:
        # Define column names
        index_names = ['Engine', 'Cycle']
        setting_names = ['Setting 1', 'Setting 2', 'Setting 3']
        sensor_names = ['Fan Inlet Temperature (◦R)',
               'LPC Outlet Temperature (◦R)',
               'HPC Outlet Temperature (◦R)',
               'LPT Outlet Temperature (◦R)',
               'Fan Inlet Pressure (psia)',
               'Bypass-Duct Pressure (psia)',
               'HPC Outlet Pressure (psia)',
               'Physical Fan Speed (rpm)',
               'Physical Core Speed (rpm)',
               'Engine Pressure Ratio (P50/P2)',
               'HPC Outlet Static Pressure (psia)',
               'Ratio of Fuel Flow to Ps30 (pps/psia)',
               'Corrected Fan Speed (rpm)',
               'Corrected Core Speed (rpm)',
               'Bypass Ratio',
               'Burner Fuel-Air Ratio',
               'Bleed Enthalpy',
               'Required Fan Speed',
               'Required Fan Conversion Speed',
               'High-Pressure Turbines Cool Air Flow',
               'Low-Pressure Turbines Cool Air Flow',
               'Sensor 26',
               'Sensor 27']

        col_names = index_names + setting_names + sensor_names

        # Read the uploaded file into a DataFrame
        df = pd.read_csv(file.file, sep='\s+', header=None, names=col_names)

        # Perform the necessary transformations on the DataFrame
        # 1. Add RUL
        feature_array_with_rul = rul_adder.transform(df)

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
