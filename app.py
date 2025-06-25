from flask import Flask, render_template, request, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
from stable_foocus_module import stableFoocus
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.secret_key = 'fooocus_secret'

# Ensure upload/result folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Accept both preset image URLs and file uploads
    upper_url = request.form.get('selected_upper_apparel')
    lower_url = request.form.get('selected_lower_apparel')
    model = request.files.get('model_image')
    if not model:
        return jsonify({'error': 'Please upload a model image.'}), 400

    timestamp = int(time.time())
    model_filename = f"model_{timestamp}_" + secure_filename(model.filename)
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
    model.save(model_path)

    # Download preset images if URLs are provided, else None
    def download_image(url, prefix):
        import requests
        from urllib.parse import urlparse
        ext = os.path.splitext(urlparse(url).path)[1] or '.jpg'
        filename = f"{prefix}_{timestamp}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        r = requests.get(url)
        with open(filepath, 'wb') as f:
            f.write(r.content)
        return filepath

    upper_path = download_image(upper_url, "upper") if upper_url else None
    lower_path = download_image(lower_url, "lower") if lower_url else None

    try:
        # Always use stableFoocus(model_path, upper_path, lower_path)
        result_url = stableFoocus(model_path, upper_path, lower_path)
    except Exception as e:
        return jsonify({'error': f'Processing failed: {e}'}), 500

    return jsonify({'result_url': result_url})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
