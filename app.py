from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the model
model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

@app.route('/caption', methods=['POST'])
def captioner():
    # Check if an image is provided
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get the image file
    file = request.files['image']
    
    # Read the image
    img = Image.open(file.stream)
    
    # Generate the caption
    result = model(img)[0]['generated_text']
    
    # Return the result as JSON
    return jsonify({'caption': result})

if __name__ == '__main__':
    app.run(debug=True)
