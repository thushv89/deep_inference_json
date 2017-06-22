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

# call this method if the weight file is not found
def maybe_download(weight_url):
    raise NotImplementedError

def load_weights_from_file(weight_file):
    global logger
    global sess
    global TF_SCOPES,WEIGHTS,BIASES
    '''
    Loads the weights from a file and assign them to tensorflow variables
    :param weight_file: File name
    :return: the tensorflow operations to assign correct values to weights
    '''

    # download the weight file if not existing
    # also notify user of the size of the weight file (large)
    weights = np.load(weight_file)
    keys = sorted(weights.keys())
    var_shapes = {}
    logger.info('Printing Information about Available weights...')
    for i, k in enumerate(keys):
        logger.info('\tKey ID: %d, Key: %s, Weight shape: %s'%(i, k, list(np.shape(weights[k]))))
        var_shapes[k] = list(np.shape(weights[k]))

    logger.debug('Variable shapes dictionary')
    logger.debug('%s\n',var_shapes)
    build_vgg_variables(var_shapes)

    for si,scope in enumerate(TF_SCOPES):
        logger.debug('\tAssigining values for scope %s',scope)
        with tf.variable_scope(scope,reuse=True):
            weight_key, bias_key = scope + '_W', scope + '_b'
            sess.run(tf.assign(tf.get_variable(TF_WEIGHTS_STR),weights[weight_key]))
            #del weights[weight_key]
            sess.run(tf.assign(tf.get_variable(TF_BIAS_STR), weights[bias_key]))
            #del weights[bias_key]

    del weights
    #return tf_var_assign_ops

def build_vgg_variables(variable_shapes):
    '''
    Build the required tensorflow variables to populate the VGG-16 model
    All are initialized with zeros (initialization doesn't matter in this case as we assign exact values later)
    :param variable_shapes: Shapes of the variables
    :return:
    '''
    global logger
    global TF_SCOPES,WEIGHTS,BIASES

    logger.info("Building VGG Variables (Tensorflow)...")
    for si,scope in enumerate(TF_SCOPES):
        with tf.variable_scope(scope):
            weight_key, bias_key = TF_SCOPES[si]+'_W', TF_SCOPES[si]+'_b'
            weights = tf.get_variable(TF_WEIGHTS_STR, variable_shapes[weight_key],
                                      initializer=tf.constant_initializer(0.0))
            bias = tf.get_variable(TF_BIAS_STR, variable_shapes[bias_key],
                                   initializer = tf.constant_initializer(0.0))
            WEIGHTS[TF_SCOPES[si]], BIASES[TF_SCOPES[si]] = weights, bias


def infererence(tf_inputs):
    global logger
    global TF_SCOPES, TF_WEIGHTS_STR, TF_BIAS_STR, MAX_POOL_INDICES

    for si, scope in enumerate(TF_SCOPES):
        with tf.variable_scope(scope,reuse=True):
            weight, bias = tf.get_variable(TF_WEIGHTS_STR), tf.get_variable(TF_BIAS_STR)

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

            if si in MAX_POOL_INDICES:
                h = tf.nn.max_pool(h,[1,2,2,1],[1,2,2,1],padding='SAME',name='hidden')

    return out


def preprocess_inputs(filenames):

    with tf.name_scope('preprocess'):
        # FIFO Queue of file names
        filename_queue = tf.train.string_input_producer(filenames, capacity=10)

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
        image_batch = tf.train.batch([std_image], batch_size = 1, capacity = 10, name='image_batch')

        # to use record reader we need to use a Queue either random

    print('Preprocessing done')
    return image_batch

logging_level = logging.DEBUG
logging_format = '[%(name)s] [%(funcName)s] %(message)s'

sess = None
TF_SCOPES = ['conv1_1','conv1_2','conv2_1','conv2_2','conv3_1','conv3_2','conv3_3',
             'conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3',
             'fc6','fc7','fc8']
MAX_POOL_INDICES = [1,3,6,9,12]

TF_WEIGHTS_STR = 'weights'
TF_BIAS_STR = 'bias'

WEIGHTS = {}
BIASES = {}

if __name__=='__main__':

    logger = logging.getLogger('Logger')
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(logging_format))
    console.setLevel(logging_level)
    logger.addHandler(console)

    with tf.Session() as sess:

        tf.global_variables_initializer().run()
        load_weights_from_file('vgg16_weights.npz')

        image_height,image_width,image_depth = 224,224,3

        batch_size = 1

        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(coord=coord)

        # creates a FIFO queue until the reader needs them

        filenames = ['cat.jpg', 'dog.jpg','scorpion.jpg']

        # ================================== VERY IMPORTANT ==================================
        # Defining the coordinator and startring queue runner should happen ONLY AFTER you define your queues
        # i.e. preprocess_inputs(...) Otherwise the process will hang forever
        # https://stackoverflow.com/questions/35274405/tensorflow-training-using-input-queue-gets-stuck
        tf_images = preprocess_inputs(filenames)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        tf_prediction = infererence(tf_images)
        for i in range(3):
            #images = sess.run(tf_images)
            pred = sess.run(tf_prediction)

            logger.info('Class: %d (%s)',np.argmax(pred),imagenet_classes.class_names[np.argmax(pred)])
            logger.info('Confidence: %.2f',np.max(pred))

        #plt.imshow(images[0])
        #plt.show()
        print('Session ran')

        coord.request_stop()
        coord.join(threads)

        #print(images)
        #plt.imshow(images)


