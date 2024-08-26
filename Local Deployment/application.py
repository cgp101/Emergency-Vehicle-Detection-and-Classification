from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64
from datetime import datetime

application = Flask(__name__)

model_mobilenet = tf.keras.models.load_model('mobilenet_deted.h5')

def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_frame(frame):
    img_array = preprocess_image(frame)
    predictions = model_mobilenet.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = np.argmax(predictions)
    class_labels = {0: 'Normal Vehicle', 1: 'Ambulance', 2: 'Fire Truck', 3: 'Police Car'}
    return confidence, class_labels[predicted_class]

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Get the current time for prediction
    prediction_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Open the image and convert to RGB
    img = Image.open(file.stream).convert('RGB')
    
    # Perform the prediction
    confidence_score, predicted_class = predict_frame(img)

    # Convert the original image to a format that can be displayed on the webpage
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    img_data = base64.b64encode(img_io.getvalue()).decode()

    # Return the results as JSON
    return jsonify({
        'image_data': img_data,
        'predicted_class': predicted_class,
        'confidence_score': f"{confidence_score:.2f}",
        'prediction_time': prediction_time
    })

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5001, debug=True)
