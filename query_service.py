from urllib.error import URLError

from flask import Flask, jsonify
import json
import urllib
import vgg_inference
from PIL import Image
import os

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"


@app.route('/infer/<int:batch_size>', methods=['GET'])
def get_urls(batch_size):
    with open('input_url.json') as data_file:
        data = json.load(data_file)
        url_array = data['image-urls']
        status_array = []
        filenames = []
        for url in url_array:
            info_dict = download_image(url)
            status_array.append(info_dict)
            filenames.append(info_dict['saved_files'])

        pred,conf = infer_vgg(filenames,batch_size)
        return jsonify({'download_status': status_array, 'inference_status': [pred,conf]})

image_index = 0
infer_filenames = []


def download_image(url):
    global image_index
    error = 'None'
    success = False
    file_extension = url.split('.')[-1].lower()
    downloaded_files = []
    if file_extension not in ['jpg','jpeg','png']:
        error = 'Please provide urls of images only of JPG(JPEG) or PNG format (%s)'%(url)
    else:
        try:
            # avoiding saving files as JPEG
            if file_extension=='jpeg':
                file_extension = 'jpg'

            urllib.request.urlretrieve(url, "image-%d.%s"%(image_index,file_extension))
        except urllib.error.HTTPError as e1:
            error = 'A HTTP Error occured (%d)'%e1.code
        except urllib.error.URLError as e2:
            error = 'A URL error occured (%s)'%e2.reason

        save_filename = "image-%d.%s"%(image_index,file_extension)

        # converting png files to jpg
        if file_extension == 'png':
            im = Image.open(save_filename)
            im = im.convert('RGB')
            fname,fext = os.path.splitext(save_filename)
            save_filename = fname+'.jpg'
            im.save(save_filename)

        image_index += 1
        print('Downloaded image')
        success = True

    return {'url':url, 'success_status':success, 'error':error, 'saved_files': save_filename}

def infer_vgg(filenames,batch_size):
    prediction_list, confidence_list = vgg_inference.infer_from_vgg(filenames,batch_size)
    return prediction_list, confidence_list

if __name__ == '__main__':
    app.run(debug=True)