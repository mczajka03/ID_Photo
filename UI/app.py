from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import os

from Evaluate.nn_evaluate import type_of_photo

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def predict_by_nn(file_path):
    prediction = type_of_photo(file_path)

    return prediction


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Upload and Process Image</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .drop-zone {
                width: 300px; height: 200px; border: 2px dashed #ccc; 
                border-radius: 10px; display: flex; justify-content: center; 
                align-items: center; margin: 20px auto; color: #aaa; cursor: pointer;
                position: relative; overflow: hidden;
            }
            .drop-zone img {
                position: absolute;
                top: 0; left: 0; width: 100%; height: 100%;
                object-fit: contain; display: none;
            }
            .drop-zone span {
                z-index: 1;
            }
            .loading {
                display: none;
                text-align: center;
                margin-top: 20px;
                font-size: 18px;
            }
            .loading span {
                display: inline-block;
                width: 25px;
                height: 25px;
                border: 3px solid #007bff;
                border-top-color: transparent;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            button {
                display: block;
                margin: 20px auto;
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                background-color: #007bff;
                color: white;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover { background-color: #0056b3; }
            #output { text-align: center; margin-top: 20px; font-size: 18px; }
        </style>
    </head>
    <body>
        <h1 style="text-align: center;">Upload and Process Image</h1>
        <div 
            class="drop-zone" id="drop-zone" 
            ondrop="handleDrop(event)" 
            ondragover="handleDragOver(event)" 
            ondragleave="handleDragLeave(event)">
            <span id="drop-zone-text">Drag and drop an image here</span>
            <img id="preview" src="" alt="Uploaded image">
        </div>
        <button id="process-button" onclick="processImage()">Process</button>
        <div class="loading" id="loading">
            <span></span> Processing...
        </div>
        <div id="output"></div>
        <script>
            const dropZone = document.getElementById('drop-zone');
            const preview = document.getElementById('preview');
            const dropZoneText = document.getElementById('drop-zone-text');
            const output = document.getElementById('output');
            const loading = document.getElementById('loading');
            let uploadedFilePath = "";

            function handleDragOver(event) {
                event.preventDefault();
                dropZone.classList.add('dragover');
            }

            function handleDragLeave(event) {
                event.preventDefault();
                dropZone.classList.remove('dragover');
            }

            function handleDrop(event) {
                event.preventDefault();
                dropZone.classList.remove('dragover');
                const files = event.dataTransfer.files;
                if (files.length > 0) uploadFile(files[0]);
            }

            function uploadFile(file) {
                const formData = new FormData();
                formData.append('file', file);

                fetch('/upload', { method: 'POST', body: formData })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        uploadedFilePath = data.file_path;
                        preview.src = data.file_path;
                        preview.style.display = 'block';
                        dropZoneText.style.display = 'none';
                        output.textContent = "";
                    } else {
                        output.textContent = "Error: " + data.error;
                    }
                })
                .catch(err => {
                    output.textContent = "Error: " + err.message;
                });
            }

            function processImage() {
                if (!uploadedFilePath) {
                    output.textContent = "No image uploaded.";
                    return;
                }
                loading.style.display = 'block';
                output.textContent = "";

                fetch('/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ file_path: uploadedFilePath })
                })
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    if (data.success) {
                        output.textContent = `Prediction: ${data.result}`;
                    } else {
                        output.textContent = "Error: " + data.error;
                    }
                })
                .catch(err => {
                    loading.style.display = 'none';
                    output.textContent = "Error: " + err.message;
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return jsonify({'success': True, 'file_path': f"/uploads/{filename}"})


@app.route('/process', methods=['POST'])
def process_image():
    data = request.get_json()
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(data.get('file_path')))
    if not file_path or not os.path.exists(file_path):
        return jsonify({'success': False, 'error': 'File not found'})
    prediction = predict_by_nn(file_path)
    label_map = {0: "no glasses", 1: "regular glasses", 2: "sunglasses"}
    result = label_map[prediction]
    return jsonify({'success': True, 'result': result})


if __name__ == '__main__':
    app.run(debug=True)
