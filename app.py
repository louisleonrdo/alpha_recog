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

romaji_dict = ['a', 'ae', 'b', 'bb', 'ch', 'd', 'e', 'eo', 'eu', 'g', 'gg', 'h', 'i', 'j', 'k', 'm', 'n', 'ng', 'o', 'p', 'r', 's', 'ss', 't', 'u', 'ya', 'yae', 'ye', 'yo', 'yu']
# Load the trained model
model = load_model('model.keras')
def label_mapping(answer):
    return romaji_dict[answer]

class_mapping = {}
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for i in range(len(alphabets)):
    class_mapping[i] = alphabets[i]

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(-1, 28, 28, 1)
    return image


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

        # print(img)
        # print(data)
        # pixel_array = np.zeros(28*28)  # Initialize array
        # image = np.array(image, dtype=np.float32)
        # image = image.reshape(28, 28) / 255.0  # Normalize pixel values
        # image = image.reshape(-1, 28, 28, 1)  # Reshape to match model input

        input_data = np.reshape(image,(28, 28, 1))  # Adjust shape as needed
        # print(image)
        input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension at axis 0
        print(input_data)
        # Make a prediction
        prediction = model.predict(input_data)
        predicted_label = np.argmax(prediction, axis=1)[0]
        label = label_mapping(predicted_label)
        print(label)

        # pred = class_mapping[int(np.argmax(model.predict(image)))]
        # print(pred)
        # return jsonify({'prediction': data, 'label': pred})
        # return jsonify({'data': 'x'})
        # predicted_label = np.argmax(prediction, axis=1)[0]
        # print(predicted_label, ' ', prediction)
        # label = class_mapping.get(predicted_label, "Unknown")

        return jsonify({'prediction': label})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
