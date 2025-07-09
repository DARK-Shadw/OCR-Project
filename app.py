import easyocr
from flask import Flask, request, jsonify
import numpy as np
import cv2
import pytesseract
from PIL import Image

app = Flask(__name__)
reader = easyocr.Reader(['en'])



@app.route('/easyocr', methods=['POST'])
def easyocr_image():
    if 'images' not in request.files:
        return jsonify({'error': 'No image files provided'}), 400

    images = request.files.getlist('images')
    extracted_texts = []

    for image_file in images:
        try:
            # Read the image file
            image_np = np.frombuffer(image_file.read(), np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            # Perform OCR
            result = reader.readtext(image)
            
            # Extract text from the result
            text = " ".join([item[1] for item in result])
            extracted_texts.append(text)
        except Exception as e:
            extracted_texts.append(f"Error processing image with EasyOCR: {str(e)}")

    return jsonify({'extracted_texts': extracted_texts})

@app.route('/tesseract', methods=['POST'])
def tesseract_image():
    if 'images' not in request.files:
        return jsonify({'error': 'No image files provided'}), 400

    images = request.files.getlist('images')
    extracted_texts = []

    for image_file in images:
        try:
            # Read the image file and convert to PIL Image
            image = Image.open(image_file.stream)

            # Perform OCR using Tesseract
            text = pytesseract.image_to_string(image)
            extracted_texts.append(text.strip())
        except Exception as e:
            extracted_texts.append(f"Error processing image with Tesseract: {str(e)}")

    return jsonify({'extracted_texts': extracted_texts})

if __name__ == '__main__':
    app.run(debug=True)