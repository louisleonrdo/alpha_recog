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
from keras.optimizers import Adam

import numpy as np


app = Flask(__name__)
CORS(app)
# custom_objects = {'Adam': Adam}

model = load_model('letter_recognition.keras')


# def label_mapping(answer):
#     return alphabets[answer]

# alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# for i in range(len(alphabets)):
#     class_mapping[i] = alphabets[i]

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(-1, 28, 28, 1)
    return image

class_mapping = {}
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for i in range(len(alphabets)):
    class_mapping[i] = alphabets[i]
class_mapping

@app.route('/predict', methods=['post'])
def predict():
    try:
        # alphabet        
        data = request.get_json(force=True)
        # if 'image' not in data:
        #     return jsonify({'error': 'No image data provided'}), 400

        # image_data = base64.b64decode(data['image'])
        # img = preprocess_image(image_data)

        # hangul
        image = data['pixelData']
        print(image)
        # print(img)
        # print(data)
        # pixel_array = np.zeros(28*28)  # Initialize array
        image = np.array(image, dtype=np.float32)
        # image = image.reshape(-1, 28, 28, 1)  # Reshape to match model input
        print(image.shape)
        # print(image)
        # image = image.reshape(28, 28) / 255.0  # Normalize pixel values
        pred = class_mapping[int(np.argmax(model.predict(image)))]

        # input_data = np.reshape(image,(28, 28, 1))  # Adjust shape as needed
        # # print(image)
        # input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension at axis 0
        # print(input_data)
      
      
        # Make a prediction
        # prediction = model.predict(input_data)
        # predicted_label = np.argmax(prediction, axis=1)[0]
        # label = label_mapping(predicted_label)
        print(pred)

        return jsonify({'prediction': pred})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
