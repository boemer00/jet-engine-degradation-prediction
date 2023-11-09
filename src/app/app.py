# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np

# # Initialise the Flask application
# app = Flask(__name__)

# # Load the trained model
# model = tf.keras.models.load_model('/jet-engine/model.keras')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Parse JSON data coming from the request
#         data = request.get_json()
#         # Assume data['features'] is a list of features
#         features = np.array(data['features'])

#         # Perform any necessary reshaping, preprocessing here
#         # For example, if your model expects data of a certain shape
#         # features = features.reshape(expected_shape)

#         # Generate predictions
#         predictions = model.predict(features)

#         # Convert predictions to a list for JSON response
#         predictions_list = predictions.tolist()

#         # Send back the predictions as a response
#         return jsonify(predictions=predictions_list)

#     except Exception as e:
#         return jsonify(error=str(e)), 500

# if __name__ == '__main__':
#     # Run the Flask app
#     app.run(host='0.0.0.0', port=5000)
