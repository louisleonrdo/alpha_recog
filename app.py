from flask import Flask, request, jsonify
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from flask_cors import CORS

from keras.models import load_model
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
model = load_model('model.keras')

@app.route('/predict', methods=['post'])
def predict():
    try:
        data = request.get_json(force=True)
        image = data['pixelData']


        # pixel_array = np.zeros(28*28)  # Initialize array
        # for key, value in image.items():
        #     pixel_array[int(key)] = 1 - value

        input_data = np.reshape(image,(28, 28, 1))  # Adjust shape as needed
        print(input_data)
        # print(input_data[0])
        print(input_data.shape)
        # Make a prediction
        prediction = model.predict(input_data)
        predicted_label = np.argmax(prediction, axis=1)[0]
        # # # gray_pixels = data.get('data', [])
        print(predicted_label)
        return jsonify({'prediction': data })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
