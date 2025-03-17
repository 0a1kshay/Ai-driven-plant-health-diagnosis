import os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import pandas as pd
from werkzeug.utils import secure_filename

# Load CSV data
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load and prepare the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device)
model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Prediction function
def predict(image_path):
    try:
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()

        return predicted_class
    except Exception as e:
        return str(e)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        try:
            # Check if an image is uploaded
            if 'image' not in request.files:
                return jsonify({'error': 'No file part'})

            image = request.files['image']
            if image.filename == '' or not allowed_file(image.filename):
                return jsonify({'error': 'Invalid file type'})

            filename = secure_filename(image.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(file_path)

            # Predict the disease
            pred = predict(file_path)

            # Check if the prediction index is valid
            if isinstance(pred, str) or pred < 0 or pred >= len(disease_info):
                return jsonify({'error': 'Prediction index out of range'})

            # Fetch disease and supplement information
            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]

            return render_template('submit.html', title=title, desc=description, prevent=prevent,
                                   image_url=image_url, pred=pred, sname=supplement_name, 
                                   simage=supplement_image_url, buy_link=supplement_buy_link)
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/market', methods=['GET', 'POST'])
def market():
    try:
        return render_template('market.html', 
                               supplement_image=list(supplement_info['supplement image']),
                               supplement_name=list(supplement_info['supplement name']), 
                               disease=list(disease_info['disease_name']),
                               buy=list(supplement_info['buy link']))
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
