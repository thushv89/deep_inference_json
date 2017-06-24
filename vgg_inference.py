# This script will be using the weights of a deep convolution model (VGG-16) pretrained on a large image dataset (ILSVRC)
# The exact details are found in the paper https://arxiv.org/pdf/1409.1556.pdf
# The weights are presented in .npz (numpy) format and will be using tensorflow to develop the inference module
# The weights file is available at https://www.cs.toronto.edu/~frossard/post/vgg16/

# We will be using the Model D from the paper
# Architecture Details
# conv3-64 -> conv3-64 -> maxpool -> conv3-128 -> conv3-128 -> maxpool -> conv3-256 -> conv3-256 -> conv3-256 -> conv3-512 -> conv3-512 -> conv3-512
#                                                                                                                                              |
#                                                                                                                                              V
#                                             softmax <- FC-1000 <- FC-4096 <- FC-4096 <- maxpool <- conv3-512 <- conv3-512 <- conv3-512 <- maxpool

# Inputs should be resized to 224 x 224

import numpy as np
import tensorflow as tf
import logging
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imagenet_classes
from math import ceil
import os
import urllib
from PIL import Image
import config

ops_created = False

logging_level = logging.DEBUG
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

logger = logging.getLogger('Logger')
logger.setLevel(logging.DEBUG)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(logging.Formatter(logging_format))
console.setLevel(logging_level)
fileHandler = logging.FileHandler('main.log', mode='w')
fileHandler.setFormatter(logging.Formatter(logging_format))
fileHandler.setLevel(logging.DEBUG)
logger.addHandler(console)
logger.addHandler(fileHandler)

graph = tf.get_default_graph()
sess = tf.InteractiveSession(graph=graph)

# call this method if the weight file is not found
def maybe_download(weight_filename):
    '''
    Download the weights file if required
    :param weight_filename:
    :return:
    '''
    global logger

    if not os.path.exists(weight_filename):
        filename, _ = urllib.request.urlretrieve(config.WEIGHTS_URL, weight_filename)
        logger.warning("The file exceeds 500MB in size. But is a necessity")
    else:
        logger.info('Found the weights file locally. No need to download')

    statinfo = os.stat(weight_filename)
    if statinfo.st_size > config.WEIGHTS_FILESIZE_BYTES:
        logger.info('Found and verified %s' % weight_filename)
    else:
        logger.info('File size: %d'%statinfo.st_size)
        logger.info('Failed to verify ' + weight_filename +
                    '. Can you get to it with a browser? %s'%'https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz')
        raise Exception(
            'Failed to verify ' + weight_filename + '. Can you get to it with a browser? %s'%'https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz')


def load_weights_from_file(weight_file):
    '''
    Loads the weights from a file and assign them to tensorflow variables
    :param weight_file: File name
    :return: the tensorflow operations to assign correct values to weights
    '''

    global logger, sess, graph

    # download the weight file if not existing
    # also notify user of the size of the weight file (large)
    maybe_download(weight_file)
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    var_shapes = {}
    logger.info('Printing Information about Available weights...')
    for i, k in enumerate(keys):
        logger.info('\tKey ID: %d, Key: %s, Weight shape: %s'%(i, k, list(np.shape(weights[k]))))
        var_shapes[k] = list(np.shape(weights[k]))
    logger.debug('')

    logger.debug('Variable shapes dictionary')
    logger.debug('\t%s\n',var_shapes)
    build_vgg_variables(var_shapes)

    with sess.as_default() and graph.as_default():

        for si,scope in enumerate(config.TF_SCOPES):
            logger.debug('\tAssigining values for scope %s',scope)
            with tf.variable_scope(scope,reuse=True):
                weight_key, bias_key = scope + '_W', scope + '_b'
                tf_cond_weight_op = tf.cond(tf.reduce_all(tf.not_equal(tf.get_variable(config.TF_WEIGHTS_STR),tf.zeros(var_shapes[weight_key],dtype=tf.float32))),
                                            lambda: tf.constant(-1,dtype=tf.float32),
                                            lambda: tf.assign(tf.get_variable(config.TF_WEIGHTS_STR),weights[weight_key]),name='assign_weights_op')

                tf_cond_bias_op = tf.cond(tf.reduce_all(tf.not_equal(tf.get_variable(config.TF_BIAS_STR),tf.zeros(var_shapes[bias_key],dtype=tf.float32))),
                                          lambda: tf.constant(-1,dtype=tf.float32),
                                          lambda: tf.assign(tf.get_variable(config.TF_BIAS_STR), weights[bias_key],name='assign_bias_op'))

                _ = sess.run([tf_cond_weight_op,tf_cond_bias_op])
                _ = sess.run([])

            # Can used to debug the counts of operations variables created
            #op_count = len(graph.get_operations())
            #var_count = len(tf.global_variables()) + len(tf.local_variables()) + len(tf.model_variables())
            #print(op_count,var_count)
        logger.debug('')
    del weights


def build_vgg_variables(variable_shapes):
    '''
    Build the required tensorflow variables to populate the VGG-16 model
    All are initialized with zeros (initialization doesn't matter in this case as we assign exact values later)
    :param variable_shapes: Shapes of the variables
    :return:
    '''
    global logger,sess,graph

    logger.info("Building VGG Variables (Tensorflow)...")
    with sess.as_default and graph.as_default():
        for si,scope in enumerate(config.TF_SCOPES):
            with tf.variable_scope(scope) as sc:
                weight_key, bias_key = config.TF_SCOPES[si]+'_W', config.TF_SCOPES[si]+'_b'

                # Try Except because if you try get_variable with an intializer and
                # the variable exists, you will get a ValueError saying the variable exists
                #
                try:
                    weights = tf.get_variable(config.TF_WEIGHTS_STR, variable_shapes[weight_key],
                                              initializer=tf.constant_initializer(0.0))
                    bias = tf.get_variable(config.TF_BIAS_STR, variable_shapes[bias_key],
                                           initializer = tf.constant_initializer(0.0))

                    sess.run(tf.variables_initializer([weights,bias]))

                except ValueError:
                    logger.debug('Variables in scope %s already initialized\n'%scope)


def infererence(tf_inputs):
    '''
    Inferencing the model. The input (tf_inputs) is propagated through convolutions poolings
    fully-connected layers to obtain the final softmax output
    :param tf_inputs: a batch of images (tensorflow placeholder)
    :return:
    '''
    global logger

    for si, scope in enumerate(config.TF_SCOPES):
        with tf.variable_scope(scope,reuse=True) as sc:
            weight, bias = tf.get_variable(config.TF_WEIGHTS_STR), tf.get_variable(config.TF_BIAS_STR)

            if 'fc' not in scope:
                if si == 0:
                    h = tf.nn.relu(tf.nn.conv2d(tf_inputs,weight,strides=[1,1,1,1],padding='SAME')+bias,name='hidden')
                else:
                    h = tf.nn.relu(tf.nn.conv2d(h, weight, strides=[1, 1, 1, 1], padding='SAME') + bias,
                                   name='hidden')
            else:
                # Reshaping required for the first fulcon layer
                if scope == 'fc6':
                    h_shape = h.get_shape().as_list()
                    h = tf.reshape(h,[h_shape[0], h_shape[1] * h_shape[2] * h_shape[3]])

                if scope == 'fc8':
                    out = tf.nn.softmax(tf.matmul(h,weight) + bias, name='output')
                else:
                    h = tf.nn.relu(tf.matmul(h, weight) + bias, name= 'hidden')

            if si in config.MAX_POOL_INDICES:
                h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME',name='hidden')

    return out


def preprocess_inputs_with_tfqueue(filenames, batch_size):
    '''
    An advance input pipeline implemented with tensorflow. However this is
    quite ineffective for the given problem. So not using this at the moment
    # ================================== VERY IMPORTANT ==================================
        # Defining the coordinator and startring queue runner should happen ONLY AFTER you define your queues
        # i.e. preprocess_inputs(...) Otherwise the process will hang forever
        # https://stackoverflow.com/questions/35274405/tensorflow-training-using-input-queue-gets-stuck
    # ================================= EXAMPLE CODE FOR USING QUEUES =========================================
        #tf_images = preprocess_inputs(filenames,batch_size)
        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        ... run training
        #coord.request_stop()
        #coord.join(threads)
    :param filenames: filenames
    :param batch_size: the size of a single batch that should be returned
    :return:
    '''
    global sess,graph
    logger.info('Received filenames: %s',filenames)
    with sess.as_default() and graph.as_default() and tf.name_scope('preprocess'):
        # FIFO Queue of file names
        # creates a FIFO queue until the reader needs them
        filename_queue = tf.train.string_input_producer(filenames, capacity=10, shuffle=False)

        # Reader which takes a filename queue and read() which outputs data one by one
        reader = tf.WholeFileReader()
        _, image_buffer = reader.read(filename_queue, name='image_read_op')

        # return uint8
        dec_image = tf.image.decode_jpeg(contents=image_buffer,channels=3,name='decode_jpg')
        # convert to float32
        float_image = tf.image.convert_image_dtype(dec_image,dtype=tf.float32,name= 'float_image')
        # resize image
        resized_image = tf.image.resize_images(float_image,[224,224])
        # standardize image
        std_image = tf.image.per_image_standardization(resized_image)
        # https://stackoverflow.com/questions/37126108/how-to-read-data-into-tensorflow-batches-from-example-queue

        # The batching mechanism that takes a output produced by reader (with preprocessing) and outputs a batch tensor
        # [batch_size, height, width, depth] 4D tensor
        image_batch = tf.train.batch([std_image], batch_size = batch_size, capacity = 10, name='image_batch')

        # to use record reader we need to use a Queue either random

    print('Preprocessing done\n')
    return image_batch


def preprocess_inputs_with_pil(filenames):
    '''
    Pre process images given in a set of filenames
    :param filenames: names of the files
    :return:
    '''
    image_batch = None
    logger.debug('Preprocessing all images')
    for fn in filenames:
        logger.debug('\tProcessing %s'%fn)
        im = Image.open(fn)
        logger.debug('\tOriginal size %s: ',np.asarray(im).shape)
        # the model processes images of size 224, 224, 3
        # so all images need to be resized to that size
        im.thumbnail((config.RESIZE_SIDE, config.RESIZE_SIDE), Image.ANTIALIAS)

        im_arr = np.asarray(im, dtype=np.float32)
        im_arr = (im_arr - np.mean(im_arr)) / np.std(im_arr)

        im_shape = im_arr.shape

        # If  the image width and height is below 224 pixels
        if im_shape[0] < config.RESIZE_SIDE:
            im_arr = np.append(im_arr, np.zeros((config.RESIZE_SIDE - im_shape[0], im_shape[1], 3), dtype=np.float32), axis=0)
            im_shape = im_arr.shape
        if im_shape[1] < config.RESIZE_SIDE:
            im_arr = np.append(im_arr, np.zeros((im_shape[0], config.RESIZE_SIDE - im_shape[1], 3), dtype=np.float32), axis=1)
        logger.debug('\tSize after resizing: %s\n',im_arr.shape)

        # Creating an image batch using the preprocessed images
        if image_batch is None:
            image_batch = np.reshape(im_arr, (1, config.RESIZE_SIDE, config.RESIZE_SIDE, 3))
        else:
            image_batch = np.append(image_batch, np.reshape(im_arr, (1, config.RESIZE_SIDE, config.RESIZE_SIDE, 3)), axis=0)

    logger.debug('Created an image batch of size: %s\n',image_batch.shape)
    return image_batch


def infer_from_vgg(filenames,confidence_threshold = None):
    global sess,graph,logger, ops_created
    logger.info('Recieved filenames from webservice: %s\n',filenames)

    with sess.as_default() and graph.as_default():

        if not ops_created:
            logger.info('Loading weights...\n')
            load_weights_from_file(config.WEIGHTS_FILENAME)
            ops_created = True

        image_batch = preprocess_inputs_with_pil(filenames)

        tf_inputs = tf.placeholder(shape=[len(filenames),config.RESIZE_SIDE,config.RESIZE_SIDE,3],dtype=tf.float32)
        tf_prediction = infererence(tf_inputs)

        ## TODO: Provide the top 5 classes
        prediction_list, top_5_list, confidence_list = [],[],[]

        pred = sess.run(tf_prediction,feed_dict={tf_inputs:image_batch})
        prediction_list.extend(list(np.argmax(pred, axis=1)))
        confidence_list.extend(list(np.max(pred, axis=1)))

        logger.debug('Results summary')
        for fn,p,c in zip(filenames,prediction_list,confidence_list):
            logger.debug('\tFilename: %s, Predicted class: %s, Confidence: %.4f',fn,imagenet_classes.class_names[p],c)

        if confidence_threshold is not None:
            logger.debug('Returning the results only with a confidence higher than %.5f',confidence_threshold)
            selected_input_ind = np.where(np.asarray(confidence_list)>confidence_threshold)[0]
            selected_input_ind = list(selected_input_ind.reshape(-1))
            logger.debug('Selected indices (> confidence threshold) %s',selected_input_ind)
            prediction_list = [prediction_list[pp] for pp in selected_input_ind]
            confidence_list = [confidence_list[pp] for pp in selected_input_ind]

        fname_pred_class_list = [imagenet_classes.class_names[pred] + ' (Save filename: ' + fname + ')' for fname,pred in zip(filenames,prediction_list)]
        confidence_list = [float(ceil(conf*10000)/10000) for conf in confidence_list] # rounding to 4 decimal places

        logger.info('Session finished\n')

        return fname_pred_class_list,confidence_list


def get_weight_parameter_with_key(key,weights_or_bias):
    global sess
    with sess.as_default() and graph.as_default():
        with tf.variable_scope(key,reuse=True) as sc:
            if weights_or_bias == 'weights':
                return sess.run(tf.get_variable(config.TF_WEIGHTS_STR))
            elif weights_or_bias =='bias':
                return sess.run(tf.get_variable(config.TF_BIAS_STR))
            else:
                raise NotImplementedError
