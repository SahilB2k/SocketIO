# # app.py
# import os
# import torch
# import torch.nn as nn
# from flask import Flask, request, jsonify, render_template_string, redirect, url_for
# import io
# from PIL import Image
# import torchvision.transforms as transforms
# import base64

# app = Flask(__name__)

# # Define your model architecture here
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         # Example architecture - replace with your actual model definition
#         self.main = nn.Sequential(
#             nn.Conv2d(3, 64, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(64, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(128, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(256, 512, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
            
#             nn.Conv2d(512, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()

#         )

#     def forward(self, input):
#         return self.main(input).view(-1, 1).squeeze(1)

# # Add your model to the safe globals list
# torch.serialization.add_safe_globals([Discriminator])

# # Load model
# def load_model():
#     try:
#         # Direct model loading since the saved file is the entire model
#         model = torch.load('discriminator.pth', map_location=torch.device('cpu'), weights_only=False)
#         model.eval()
#         return model
#     except Exception as e:
#         print(f"Error loading model directly: {e}")
        
#         # Alternative method
#         try:
#             loaded_model = torch.load('discriminator.pth', map_location=torch.device('cpu'), weights_only=False)
#             model = Discriminator()
#             model.load_state_dict(loaded_model.state_dict())
#             model.eval()
#             return model
#         except Exception as e2:
#             print(f"Error with alternative loading method: {e2}")
#             raise

# # Transform input images
# transform = transforms.Compose([
#     transforms.Resize(64),  # Adjust size according to your model's requirements
#     transforms.CenterCrop(64),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# # Initialize model at startup
# try:
#     model = load_model()
#     print("Model loaded successfully!")
# except Exception as e:
#     print(f"Failed to load model: {e}")
#     model = None  # Allow the app to run even if model loading fails

# # HTML template for the homepage
# HOMEPAGE_TEMPLATE = '''
# <!DOCTYPE html>
# <html>
# <head>
#     <title>Image Forgery Detection</title>
#     <meta name="viewport" content="width=device-width, initial-scale=1">
#     <style>
#         body {
#             font-family: Arial, sans-serif;
#             max-width: 800px;
#             margin: 0 auto;
#             padding: 20px;
#             line-height: 1.6;
#         }
#         h1 {
#             color: #333;
#         }
#         .container {
#             background-color: #f9f9f9;
#             border-radius: 5px;
#             padding: 20px;
#             margin-top: 20px;
#         }
#         .form-group {
#             margin-bottom: 15px;
#         }
#         label {
#             display: block;
#             margin-bottom: 5px;
#             font-weight: bold;
#         }
#         input[type="file"] {
#             display: block;
#             margin-bottom: 10px;
#         }
#         button {
#             background-color: #4CAF50;
#             color: white;
#             padding: 10px 15px;
#             border: none;
#             border-radius: 4px;
#             cursor: pointer;
#         }
#         button:hover {
#             background-color: #45a049;
#         }
#         .result {
#             margin-top: 20px;
#             padding: 15px;
#             border-radius: 5px;
#         }
#         .real {
#             background-color: #dff0d8;
#             border: 1px solid #d6e9c6;
#         }
#         .fake {
#             background-color: #f2dede;
#             border: 1px solid #ebccd1;
#         }
#         .preview {
#             max-width: 300px;
#             margin-top: 15px;
#         }
#         .hidden {
#             display: none;
#         }
#     </style>
# </head>
# <body>
#     <h1>Image Forgery Detection</h1>
    
#     <div class="container">
#         <form id="upload-form" enctype="multipart/form-data">
#             <div class="form-group">
#                 <label for="image">Upload an image:</label>
#                 <input type="file" id="image" name="file" accept="image/*" onchange="previewImage()">
#                 <img id="preview" class="preview hidden" src="" alt="Image preview">
#             </div>
#             <button type="submit">Analyze Image</button>
#         </form>
#     </div>
    
#     <div id="result" class="result hidden"></div>
    
#     <script>
#         function previewImage() {
#             const fileInput = document.getElementById('image');
#             const preview = document.getElementById('preview');
            
#             if (fileInput.files && fileInput.files[0]) {
#                 const reader = new FileReader();
                
#                 reader.onload = function(e) {
#                     preview.src = e.target.result;
#                     preview.classList.remove('hidden');
#                 }
                
#                 reader.readAsDataURL(fileInput.files[0]);
#             }
#         }
        
#         document.getElementById('upload-form').addEventListener('submit', function(e) {
#             e.preventDefault();
            
#             const formData = new FormData();
#             const fileInput = document.getElementById('image');
            
#             if (fileInput.files.length === 0) {
#                 alert('Please select an image first');
#                 return;
#             }
            
#             formData.append('file', fileInput.files[0]);
            
#             fetch('/predict', {
#                 method: 'POST',
#                 body: formData
#             })
#             .then(response => response.json())
#             .then(data => {
#                 const resultDiv = document.getElementById('result');
#                 resultDiv.classList.remove('hidden', 'real', 'fake');
                
#                 if (data.error) {
#                     resultDiv.textContent = 'Error: ' + data.error;
#                 } else {
#                     const score = (data.prediction * 100).toFixed(2);
#                     const isReal = data.is_real;
                    
#                     resultDiv.classList.add(isReal ? 'real' : 'fake');
#                     resultDiv.innerHTML = `
#                         <h3>Analysis Result:</h3>
#                         <p><strong>Prediction Score:</strong> ${score}%</p>
#                         <p><strong>Classification:</strong> ${isReal ? 'Likely Real' : 'Likely Fake'}</p>
#                         <p>Higher scores indicate the image is more likely to be authentic.</p>
#                     `;
#                 }
                
#                 resultDiv.scrollIntoView({ behavior: 'smooth' });
#             })
#             .catch(error => {
#                 console.error('Error:', error);
#                 alert('An error occurred during analysis. Please try again.');
#             });
#         });
#     </script>
# </body>
# </html>
# '''

# @app.route('/')
# def home():
#     # Return the HTML template for the homepage
#     return render_template_string(HOMEPAGE_TEMPLATE)

# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None:
#         return jsonify({'error': 'Model is not loaded properly'}), 500
        
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part in the request'}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No file selected for uploading'}), 400
    
#     try:
#         # Read image
#         img_bytes = file.read()
#         image = Image.open(io.BytesIO(img_bytes))
        
#         # Preprocess image
#         tensor = transform(image).unsqueeze(0)
        
#         # Make prediction
#         with torch.no_grad():
#             prediction = model(tensor).item()
        
#         return jsonify({
#             'prediction': prediction,
#             'is_real': prediction > 0.5  # Assuming threshold of 0.5
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/health', methods=['GET'])
# def health():
#     return jsonify({'status': 'ok', 'model_loaded': model is not None})

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(host='0.0.0.0', port=port, debug=True)


