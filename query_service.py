from urllib.error import URLError

from flask import Flask, jsonify, request
import json
import urllib
import vgg_inference
from PIL import Image
import os


app = Flask(__name__)

@app.route('/')
def index():
    return "<h2>Welcome to Deep Inference Service</h2>"+\
           "<p>This service allows you to query a pretrained deep network with images and get the class the image belongs to along with a confidence value. " +\
           "The model used for inference is VGG-16 and is pretrained on ILSVRC data." + \
           "The original paper of the model can be found <a href=\"https://arxiv.org/pdf/1409.1556.pdf\" target=\"_blank\">here</a>. " + \
           "The weights of the model were downloaded from <a href=\"https://www.cs.toronto.edu/~frossard/post/vgg16/\" target=\"_blank\">here</a>."

@app.route('/infer_from_url')
def get_url_from_json():
    url = request.args['url']

    infor_dict = download_image(url)
    filename = [infor_dict['saved_as']]

    pred,conf = infer_vgg(filename,1)
    result = {"url": url, "class": pred[0], "confidence": float(conf[0])}

    return jsonify({'download_status': infor_dict, 'result': result})

@app.route('/infer', methods=['GET'])
def get_urls():
    global image_index,infer_filenames
    image_index, infer_filenames = 0, []

    json_fname = request.args['filename']
    batch_size = int(request.args['batch_size'])

    with open(json_fname) as data_file:
        data = json.load(data_file)
        url_array = data['image-urls']
        status_array = []
        filenames = []
        for url in url_array:
            info_dict = download_image(url)
            status_array.append(info_dict)
            filenames.append(info_dict['saved_as'])

        pred,conf = infer_vgg(filenames,batch_size)

        results_arr = []
        for url,p,c in zip(url_array,pred,conf):
            results_arr.append({"url":url,"class":p,"confidence":c})

        return jsonify({'download_status': status_array, 'results': results_arr})

'''@app.route('/infer/', methods=['GET'])
def get_urls():
    global image_index,infer_filenames
    image_index, infer_filenames = 0, []
    batch_size = 1

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

        results_arr = []
        for url,p,c in zip(url_array,pred,conf):
            results_arr.append({"url":url,"class":p,"confidence":c})

        return jsonify({'download_status': status_array, 'results': results_arr})'''

image_index = 0
infer_filenames = []

def reset_tf_graph():
    vgg_inference.reset_tensorflow_graph()

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

    return {'url':url, 'success_status':success, 'error':error, 'saved_as': save_filename}

def infer_vgg(filenames,batch_size):
    prediction_list, confidence_list = vgg_inference.infer_from_vgg(filenames,batch_size)
    return prediction_list, confidence_list

if __name__ == '__main__':
    app.run(debug=True)