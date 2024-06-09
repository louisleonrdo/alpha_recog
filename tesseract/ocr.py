import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
import pytesseract
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/process_canvas', methods=['POST'])
def process_canvas():
    data = request.get_json()
    # print(data)
    image_data = data['image'].split(',')[1]  # Remove the data URL scheme
    image_bytes = base64.b64decode(image_data)
    # Convert the image bytes to a numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # print(image)
    # Preprocess the image for Tesseract OCR
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print(gray_image)
    _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    # print(thresh_image)
    # Use Tesseract to do OCR on the preprocessed image
    text = pytesseract.image_to_string(thresh_image)
    print(text)

    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(debug=True)
