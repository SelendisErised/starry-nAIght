import io
import os
import cv2
import json
import PIL
from PIL import Image
from werkzeug.utils import secure_filename
import torch
# import stuff for our web server
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from flask import jsonify
from utility import get_base_url, allowed_file, and_syntax

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12347
base_url = get_base_url(port)
app = Flask(__name__, static_url_path=base_url + 'static')
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg','png'])
# app = Flask(__name__)

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = torch.hub.load('custom', path='BEST.PT GOES HERE')  #  EDIT THIS

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images

    # Inference
    results = model(imgs, size=640)  # includes NMS
    return results


@app.route(base_url, methods=['GET', 'POST'])
def the_ai_post():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files.get('file')

        if not file:
            return

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('results',
                                    filename=filename))


@app.route(base_url + "/")
def hello_world():
    '''
    Purpose: render the home page of our website.
    Arguments: N/A
    Returns: HMTL string
    '''
    return render_template('index.html')

@app.route(base_url + "/uploads/<filename>")
def results(filename):
    '''
    Purpose: render the model's page of our website.
    Arguments: N/A
    Returns: HMTL string
    '''
    here = os.getcwd()
    image_path = os.path.join(here, app.config['UPLOAD_FOLDER'] , filename)

    image = model(image_path)
    image.print()

    if image_path.find("jpeg") != -1:
        filename = filename.replace(".jpeg",".jpg")
        image.save(os.path.join(app.config['UPLOAD_FOLDER']))

    return render_template('index.html', filename=filename)

@app.route(base_url + "/uploads/<filename>", methods=['GET', 'POST'])
def results_post(filename):
    '''
    Purpose: render the model's page of our website.
    Arguments: N/A
    Returns: HMTL string
    '''
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files.get('file')

        if not file:
            return

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('index',
                                    filename=filename))



@app.route(base_url + '/files/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


#testing urls:
with app.test_request_context():
    print(url_for('hello_world'))

if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalcx.ai-camp.org to the site where you are editing this file.
    website_url = 'coding.ai-camp.dev'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    app.run(host='0.0.0.0', port=port, debug=True)
    import sys
    sys.exit(0)
    '''
    scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)
