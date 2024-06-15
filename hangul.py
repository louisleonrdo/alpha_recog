from flask import Flask, request, jsonify
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow import keras
from flask_cors import CORS
from PIL import Image
import base64
import io


from keras.models import load_model
import numpy as np

app = Flask(__name__)
CORS(app)

model = load_model('model.keras')

romaji_dict = ['a', 'ae', 'b', 'bb', 'ch', 'd', 'e', 'eo', 'eu', 'g', 'gg', 'h', 'i', 'j', 'k', 'm', 'n', 'ng', 'o', 'p', 'r', 's', 'ss', 't', 'u', 'ya', 'yae', 'ye', 'yo', 'yu']

def label_mapping(answer):
    return romaji_dict[answer]


@app.route('/predict', methods=['post'])
def predict():
    try:
        data = request.get_json(force=True)
        image = data['pixelData']
        input_data = np.reshape(image,(28, 28, 1))  # Adjust shape as needed
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension at axis 0

        # Make a prediction
        prediction = model.predict(input_data)
        predicted_label = np.argmax(prediction, axis=1)[0]
        label = label_mapping(predicted_label)

        return jsonify({'prediction': label})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
