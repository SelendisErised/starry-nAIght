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
from utility import get_base_url
import style_transfer_VGG19

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 23456
base_url = get_base_url(port)
app = Flask(__name__, static_url_path=base_url + 'static')
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg','png'])
# app = Flask(__name__)

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route(base_url + "/")
def hello_world():
    '''
    Purpose: render the home page of our website.
    Arguments: N/A
    Returns: HMTL string
    '''
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route(base_url, methods=['GET','POST'])
def style_transfer():
    content_image = request.files.get('contentimage', '')
    style_image = request.files.get('styleimage','')
    print(content_image)
    print(style_image)

    image1 = generate(content_image, style_image)

    return render_template('index.html', filename = image1)
    # ^ needs to be changed to render_template('index.html', filename = image1) or something along those lines once generation function is complete

@app.route(base_url, methods=['GET','POST'])
def cycle_gan():
    content_image = request.files.get('contentimage', '')
    style_selected = request.form('style')

    image1 = generate2(content_image, style_image)

    return image1
    # ^ needs to be changed to render_template('index.html', filename = image1) or something along those lines once generation function is complete

def generate(image1, image2):
    #GENERATE FUNCTION GOES HERE
    #SHOULD RETURN IMAGE
    style_trans = style_transfer_VGG19.generate_img(image1, image2, result_save_path)
    image = result_save_path
#     image = "SUCCESSFUL SUBMISSION OF BOTH IMAGES"
    return image

def generate2(image, style):
    '''
    Purpose: Cycle Gan generation function
    Arguments: Uploaded image and desired style
    Returns: image
    '''


@app.route(base_url + '/files/<path:filename>')
def files(filename):
    print("CORRECTLY ASKED FOR FILE")
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


#testing urls:
with app.test_request_context():
    print(url_for('hello_world'))

if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalcx.ai-camp.org to the site where you are editing this file.
    website_url = 'cocalc13.ai-camp.dev'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    app.run(host='0.0.0.0', port=port, debug=True)
#     import sys
#     sys.exit(0)
    '''
    scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)
