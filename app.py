from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
from werkzeug.utils import secure_filename
from stable_foocus_module import stableFoocus, stableFoocusOnlyhalf
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
    upper = request.files.get('upper')
    lower = request.files.get('lower')
    model = request.files.get('model')
    if not model:
        flash('Please upload a model image.')
        return redirect(url_for('home'))

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    timestamp = int(time.time())
    model_filename = f"model_{timestamp}_" + secure_filename(model.filename)
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_filename)
    model.save(model_path)

    upper_path = lower_path = None
    if upper and upper.filename:
        upper_filename = f"upper_{timestamp}_" + secure_filename(upper.filename)
        upper_path = os.path.join(app.config['UPLOAD_FOLDER'], upper_filename)
        upper.save(upper_path)
    if lower and lower.filename:
        lower_filename = f"lower_{timestamp}_" + secure_filename(lower.filename)
        lower_path = os.path.join(app.config['UPLOAD_FOLDER'], lower_filename)
        lower.save(lower_path)

    try:
        if upper_path and lower_path:
            result_url = stableFoocus(model_path, upper_path, lower_path)
        elif upper_path:
            result_url = stableFoocusOnlyhalf(model_path, upper_path)
        elif lower_path:
            result_url = stableFoocusOnlyhalf(model_path, lower_path)
        else:
            flash('Please upload at least one upper or lower cloth image.')
            return redirect(url_for('home'))
    except Exception as e:
        flash(f'Processing failed: {e}')
        return redirect(url_for('home'))

    return render_template('index.html', result_url=result_url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
