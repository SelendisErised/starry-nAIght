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
import style_transfer_msgnet

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 12348
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
    '''
    Purpose: this is the generation function for the VGG19 Model
    Args: None
    Return: Generated image
    '''
    content_image = request.files.get('contentimage', '')
    style_image = request.files.get('styleimage','')

    here = os.getcwd()

    content_image.save(os.path.join(here, app.config['UPLOAD_FOLDER'], content_image.filename))
    style_image.save(os.path.join(here, app.config['UPLOAD_FOLDER'], style_image.filename))

    content_path = os.path.join(here, app.config['UPLOAD_FOLDER'] , content_image.filename)
    style_path = os.path.join(here, app.config['UPLOAD_FOLDER'], style_image.filename)

    save_img_path = os.path.join(here, app.config['UPLOAD_FOLDER'], "generated.jpg")
    model_path = here + '/starry-nAIght/StarryNaightFlask/21styles.model'

    style_transfer_msgnet.style_apply(content_path, style_path, save_img_path, pre_trained=model_path)

    image1 = Image.open(save_img_path)
    image1.save(os.path.join(here, 'starry-nAIght/StarryNaightFlask/static/images','generated.jpg'))

    return render_template('index.html', filename = "generated.jpg")

@app.route(base_url, methods=['GET','POST'])
def cycle_gan():
    '''
    Purpose: Cycle Gan generation function
    Arguments: Uploaded image and desired style
    Returns: image
    '''
    content_image = request.files.get('contentimage', '')
    style_selected = request.form('style')
    style = get_style(style_selected)

    os.system("python test.py --dataroot datasets/temp --name {} --model test --no_dropout --crop_size 600 --load_size 600 --aspect_ratio 0.75".format(style))



    image1 = generate2(content_image, style_image)

    return image1
    # ^ needs to be changed to render_template('index.html', filename = image1) or something along those lines once generation function is complete


def get_style(style):
    match style:
        case "Van Gogh":
            return "style_vangogh_pretrained"
        case "Monet":
            return "style_monet_pretrained"
        case "Paul Cézanne":
            return "style_cezanne_pretrained"
        case "Ukiyo-e":
            return "style_ukiyoe_pretrained"
        case _:
            return "style_vangogh_pretrained"

@app.route(base_url + '/files/<path:filename>')
def files(filename):
    print("CORRECTLY ASKED FOR FILE")
    return send_from_directory(UPLOAD_FOLDER, "generated.jpg", as_attachment=True)


#testing urls:
with app.test_request_context():
    print(url_for('hello_world'))

if __name__ == "__main__":
    '''
    coding center code
    '''
    # IMPORTANT: change the cocalcx.ai-camp.org to the site where you are editing this file.
    website_url = 'cocalc15.ai-camp.dev'
    print(f"Try to open\n\n    https://{website_url}" + base_url + '\n\n')

    app.run(host='0.0.0.0', port=port, debug=True)
#     import sys
#     sys.exit(0)
    '''
    scaffold code
    '''
    # Only for debugging while developing
    # app.run(port=80, debug=True)
