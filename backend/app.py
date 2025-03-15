# app.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
import io
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import base64
import cv2

app = Flask(__name__)

# Define your model architecture here
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Example architecture - replace with your actual model definition
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# Add forgery localization capability
class ForgeryLocalizer:
    def __init__(self, model, patch_size=64, stride=16):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def analyze_image(self, image):
        # Get original image dimensions
        orig_width, orig_height = image.size
        
        # Create a heatmap with the same dimensions as the input image
        heatmap = np.zeros((orig_height, orig_width), dtype=np.float32)
        count_map = np.zeros((orig_height, orig_width), dtype=np.float32)
        
        # Convert to RGB if it's not
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Iterate over the image in patches
        for y in range(0, orig_height - self.patch_size + 1, self.stride):
            for x in range(0, orig_width - self.patch_size + 1, self.stride):
                # Extract patch
                patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
                
                # Transform patch
                patch_tensor = self.transform(patch).unsqueeze(0)
                
                # Get model prediction
                with torch.no_grad():
                    prediction = self.model(patch_tensor).item()
                
                # Convert prediction to forgery probability (lower values mean more likely to be forged)
                forgery_prob = 1.0 - prediction
                
                # Update heatmap and count map
                heatmap[y:y+self.patch_size, x:x+self.patch_size] += forgery_prob
                count_map[y:y+self.patch_size, x:x+self.patch_size] += 1.0
        
        # Normalize heatmap by the count map (to account for overlapping patches)
        count_map = np.maximum(count_map, 1.0)  # Avoid division by zero
        heatmap = heatmap / count_map
        
        # Get overall forgery score
        overall_score = np.mean(heatmap)
        
        # Normalize heatmap for visualization
        heatmap_normalized = (heatmap * 255).astype(np.uint8)
        
        return heatmap_normalized, overall_score

    def create_overlay(self, image, heatmap, threshold=0.5):
        # Convert PIL Image to numpy array
        img_np = np.array(image)
        
        # Create colored heatmap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create a binary mask for areas above threshold
        mask = (heatmap / 255.0 > threshold).astype(np.uint8) * 255
        
        # Dilate the mask to make it more visible
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        
        # Create overlay image
        overlay = img_np.copy()
        idx = mask_dilated > 0
        
        # Apply a red tint to potentially forged regions
        overlay[idx] = overlay[idx] * 0.7 + np.array([255, 0, 0]) * 0.3
        
        # Convert back to PIL
        overlay_pil = Image.fromarray(overlay)
        heatmap_pil = Image.fromarray(heatmap_colored)
        
        return overlay_pil, heatmap_pil

# Add your model to the safe globals list
torch.serialization.add_safe_globals([Discriminator])

# Load model
def load_model():
    try:
        # Direct model loading since the saved file is the entire model
        model = torch.load('discriminator.pth', map_location=torch.device('cpu'), weights_only=False)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model directly: {e}")
        
        # Alternative method
        try:
            loaded_model = torch.load('discriminator.pth', map_location=torch.device('cpu'), weights_only=False)
            model = Discriminator()
            model.load_state_dict(loaded_model.state_dict())
            model.eval()
            return model
        except Exception as e2:
            print(f"Error with alternative loading method: {e2}")
            raise

# Transform input images for global prediction
transform = transforms.Compose([
    transforms.Resize(64),  # Adjust size according to your model's requirements
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Initialize model at startup
try:
    model = load_model()
    print("Model loaded successfully!")
    # Initialize the forgery localizer
    localizer = ForgeryLocalizer(model)
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None
    localizer = None

# Function to convert PIL Image to base64
def get_image_base64(pil_img):
    img_buffer = io.BytesIO()
    pil_img.save(img_buffer, format="PNG")
    img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"

# HTML template for the homepage
HOMEPAGE_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Image Forgery Detection with Area Localization</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: #333;
        }
        .container {
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 20px;
            margin-top: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="file"] {
            display: block;
            margin-bottom: 10px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #45a049;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 5px;
        }
        .real {
            background-color: #dff0d8;
            border: 1px solid #d6e9c6;
        }
        .fake {
            background-color: #f2dede;
            border: 1px solid #ebccd1;
        }
        .preview {
            max-width: 400px;
            margin-top: 15px;
        }
        .hidden {
            display: none;
        }
        .images-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-top: 20px;
        }
        .image-box {
            flex: 0 0 48%;
            margin-bottom: 20px;
            text-align: center;
        }
        .image-box img {
            max-width: 100%;
            border: 1px solid #ddd;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .threshold-control {
            margin-top: 15px;
            margin-bottom: 15px;
        }
        .slider-container {
            display: flex;
            align-items: center;
        }
        .slider {
            flex: 1;
            margin: 0 10px;
        }
    </style>
</head>
<body>
    <h1>Image Forgery Detection with Area Localization</h1>
    
    <div class="container">
        <form id="upload-form" enctype="multipart/form-data">
            <div class="form-group">
                <label for="image">Upload an image:</label>
                <input type="file" id="image" name="file" accept="image/*" onchange="previewImage()">
                <img id="preview" class="preview hidden" src="" alt="Image preview">
            </div>
            <button type="submit">Analyze Image</button>
        </form>
    </div>
    
    <div id="loading" class="loading">
        <div class="spinner"></div>
        <p>Analyzing image, please wait...</p>
    </div>
    
    <div id="result" class="result hidden"></div>
    
    <div id="images-container" class="images-container hidden">
        <div class="threshold-control hidden" id="threshold-control">
            <label for="threshold">Forgery Detection Threshold:</label>
            <div class="slider-container">
                <span>Less sensitive</span>
                <input type="range" min="1" max="100" value="50" class="slider" id="threshold">
                <span>More sensitive</span>
            </div>
            <button id="apply-threshold">Apply</button>
        </div>
        
        <div class="image-box">
            <h3>Original Image</h3>
            <img id="original-img" src="" alt="Original image">
        </div>
        
        <div class="image-box">
            <h3>Highlighted Forgery Areas</h3>
            <img id="overlay-img" src="" alt="Forgery areas highlighted">
        </div>
        
        <div class="image-box">
            <h3>Forgery Heatmap</h3>
            <img id="heatmap-img" src="" alt="Forgery heatmap">
        </div>
    </div>
    
    <script>
        function previewImage() {
            const fileInput = document.getElementById('image');
            const preview = document.getElementById('preview');
            
            if (fileInput.files && fileInput.files[0]) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.classList.remove('hidden');
                }
                
                reader.readAsDataURL(fileInput.files[0]);
            }
        }
        
        document.getElementById('upload-form').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const fileInput = document.getElementById('image');
            
            if (fileInput.files.length === 0) {
                alert('Please select an image first');
                return;
            }
            
            formData.append('file', fileInput.files[0]);
            
            // Show loading spinner
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').classList.add('hidden');
            document.getElementById('images-container').classList.add('hidden');
            
            fetch('/detect_forgery', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading spinner
                document.getElementById('loading').style.display = 'none';
                
                const resultDiv = document.getElementById('result');
                resultDiv.classList.remove('hidden', 'real', 'fake');
                
                if (data.error) {
                    resultDiv.textContent = 'Error: ' + data.error;
                } else {
                    const score = (data.prediction * 100).toFixed(2);
                    const isReal = data.is_real;
                    
                    resultDiv.classList.add(isReal ? 'real' : 'fake');
                    resultDiv.innerHTML = `
                        <h2>Analysis Result:</h2>
                        <p><strong>Prediction Score:</strong> ${score}%</p>
                        <p><strong>Classification:</strong> ${isReal ? 'Likely Real' : 'Likely Fake'}</p>
                        <p>Higher scores indicate the image is more likely to be authentic.</p>
                    `;
                    
                    // Display images
                    document.getElementById('original-img').src = data.original_image;
                    document.getElementById('overlay-img').src = data.overlay_image;
                    document.getElementById('heatmap-img').src = data.heatmap_image;
                    document.getElementById('images-container').classList.remove('hidden');
                    document.getElementById('threshold-control').classList.remove('hidden');
                    
                    // Store the current threshold
                    window.currentThreshold = 0.5;
                }
                
                resultDiv.scrollIntoView({ behavior: 'smooth' });
            })
            .catch(error => {
                // Hide loading spinner
                document.getElementById('loading').style.display = 'none';
                
                console.error('Error:', error);
                alert('An error occurred during analysis. Please try again.');
            });
        });  
          </script>
 </body>
 </html>
'''

@app.route('/')
def index():
    return render_template_string(HOMEPAGE_TEMPLATE)

@app.route('/detect_forgery', methods=['POST'])
def detect_forgery():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open the image
        image = Image.open(file)

        # Transform input image for global classification
        image_resized = transform(image).unsqueeze(0)

        # Predict with the model
        with torch.no_grad():
            prediction = model(image_resized).item()  # This is already a Python float

        # Classify as real or fake based on threshold
        is_real = prediction >= 0.5  # Assuming threshold of 0.5

        # Perform forgery localization
        heatmap, overall_score = localizer.analyze_image(image)
        
        # Convert numpy float32 to native Python float
        overall_score = float(overall_score)  # Convert np.float32 to Python float
        
        overlay_img, heatmap_img = localizer.create_overlay(image, heatmap)

        # Convert images to base64 for display in HTML
        overlay_base64 = get_image_base64(overlay_img)
        heatmap_base64 = get_image_base64(heatmap_img)
        original_base64 = get_image_base64(image)

        return jsonify({
            "prediction": prediction,
            "is_real": is_real,
            "overall_score": overall_score,  # Now a Python float
            "original_image": original_base64,
            "overlay_image": overlay_base64,
            "heatmap_image": heatmap_base64
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)