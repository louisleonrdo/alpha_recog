# server.py
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from model_loader import load_model

app = Flask(__name__)
model = load_model()

def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(-1, 28, 28, 1)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    image_data = base64.b64decode(data['image'])
    img = preprocess_image(image_data)
    
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    return jsonify({'prediction': chr(predicted_class + 65)})

if __name__ == '__main__':
    app.run(debug=True)
