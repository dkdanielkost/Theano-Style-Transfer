import numpy
import os
import numpy as np
import logging
from theano.tensor.signal import pool
from theano.tensor.nnet.abstract_conv import bilinear_upsampling
import joblib
from theano.tensor.nnet import conv2d
from theano.tensor.nnet import relu,softmax
import theano
import theano.tensor as T

from theano.tensor.signal.pool import pool_2d
cwd =  os.path.dirname(os.path.realpath(__file__))

'''
['convolution2d_4_weights',
 'dense_1_weights',
 'dense_2_weights',
 'convolution2d_8_weights',
 'convolution2d_5_weights',
 'convolution2d_13_weights',
 'convolution2d_7_weights',
 'convolution2d_15_weights',
 'convolution2d_59_weights',
 'convolution2d_14_weights',
 'convolution2d_16_weights',
 'dense_1_weights',
 'convolution2d_6_weights',
 'convolution2d_3_weights',
 'convolution2d_10_weights',
 'convolution2d_1_weights',
 'convolution2d_10_weights',
 'convolution2d_60_weights',
 'convolution2d_2_weights']
'''


# Read in layer weights and save to a dictionary
# Read in layer weights and save to a dictionary

cwd = os.getcwd()
direct = os.path.join(cwd,'theano_model','weights')
# weights_layer_paths = joblib.load(os.path.join(cwd,'weight_names','layer_names_weights'))
layer_weights = {}
for layer_weight_path in range(16):
#     head,layer_name = os.path.split(layer_weight_path)
    layer_weights[str(layer_weight_path) + '_w'] = joblib.load(os.path.join(direct,str(layer_weight_path) + '_w'))


# Read in bias weights and save to a dictionary
for bias_layer_path in range(16):
    layer_weights[str(bias_layer_path) + '_b'] = joblib.load(os.path.join(direct,str(bias_layer_path) + '_b'))


def drop(input, p=0.5):
    """
    :type input: numpy.array
    :param input: layer or weight matrix on which dropout is applied

    :type p: float or double between 0. and 1.
    :param p: p probability of NOT dropping out a unit, therefore (1.-p) is the drop rate.

    """
    rng = numpy.random.RandomState(1234)
    srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
    return input * mask


class DropoutHiddenLayer(object):
    def __init__(self, is_train, input, W=None, b=None,
                 activation=relu, p=0.5):
        self.input = input

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b

        self.output = activation(lin_output)
        # train_output = drop(output, p)
        # self.output = T.switch(T.neq(is_train, 0), train_output, p * output)
        self.params = [self.W, self.b]


class VGG19_conv2d_layer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, input, layer_name,image_shape,
                 activation=relu, border_mode=(2,2)):
        self.activation = activation
        self.input = input

        self.W = theano.shared(value=np.array(layer_weights[layer_name + '_w'],
                                              dtype=theano.config.floatX),
                               borrow=True)
        self.b = theano.shared(value=np.array(layer_weights[layer_name + '_b'],
                                              dtype=theano.config.floatX
                                              )
                               , borrow=True)

        self.conv_out = conv2d(
            input=input,
            input_shape=image_shape,
            filters=self.W,
            filter_shape=layer_weights[layer_name + '_w'].shape,
            border_mode=border_mode
        )

        self.output = activation(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.W, self.b]
        self.input = input

class VGG19(object):

    def __init__(self,input_image_shape,pool_method = 'average_exc_pad'):
        IMAGE_H = input_image_shape[2]
        IMAGE_W = input_image_shape[3]
        self.input = theano.tensor.tensor4('input')
        self.conv1_1 = VGG19_conv2d_layer(input=self.input,
                                    layer_name='0',
                                    image_shape=input_image_shape,
                                    border_mode=(1,1))

        self.conv1_2 = VGG19_conv2d_layer(input=self.conv1_1.output,
                                              layer_name='1',
                                              image_shape=(None, 64, IMAGE_H, IMAGE_W),
                                              border_mode=(1,1))
        self.pool1 = pool_2d(
            input = self.conv1_2.output,
            ds = (2,2),
            mode = pool_method,
            ignore_border=True,
            st = (2,2))

        self.conv2_1 = VGG19_conv2d_layer(input=self.pool1,
                                                   layer_name='2',
                                                   image_shape=(None, 64, IMAGE_H/2, IMAGE_W/2),
                                                   border_mode=(1,1))

        self.conv2_2 = VGG19_conv2d_layer(input=self.conv2_1.output,
                                                   layer_name='3',
                                                   image_shape=(None, 128, IMAGE_H/2, IMAGE_W/2),
                                                   border_mode=(1,1))

        self.pool2 = pool_2d(
            input=self.conv2_2.output,
            ds=(2, 2),
            mode=pool_method,
            ignore_border=True,
            st=(2, 2))

        self.conv3_1 = VGG19_conv2d_layer(input=self.pool2,
                                                   layer_name='4',
                                                   image_shape=(None, 128, IMAGE_H/4, IMAGE_W/4),
                                                   border_mode=(1, 1))
        self.conv3_2= VGG19_conv2d_layer(input=self.conv3_1.output,
                                                   layer_name='5',
                                                   image_shape=(None, 128, IMAGE_H/4, IMAGE_W/4),
                                                   border_mode=(1, 1))
        self.conv3_3 = VGG19_conv2d_layer(input=self.conv3_2.output,
                                                   layer_name='6',
                                                   image_shape=(None, 128, IMAGE_H/4, IMAGE_W/4),
                                                   border_mode=(1, 1))

        self.conv3_4 = VGG19_conv2d_layer(input=self.conv3_3.output,
                                                   layer_name='7',
                                                   image_shape=(None, 128, IMAGE_H/4, IMAGE_W/4),
                                                   border_mode=(1, 1))

        self.pool3 = pool_2d(
            input=self.conv3_4.output,
            ds=(2, 2),
            mode=pool_method,
            ignore_border=True,
            st=(2, 2))

        self.conv4_1 = VGG19_conv2d_layer(input=self.pool3,
                                                   layer_name='8',
                                                   image_shape=(None, 512, IMAGE_H/8, IMAGE_W/8),
                                                   border_mode=(1, 1))
        self.conv4_2 = VGG19_conv2d_layer(input=self.conv4_1.output,
                                                   layer_name='9',
                                                   image_shape=(None, 512, IMAGE_H/8, IMAGE_W/8),
                                                   border_mode=(1, 1))
        self.conv4_3 = VGG19_conv2d_layer(input=self.conv4_2.output,
                                                   layer_name='10',
                                                   image_shape=(None, 512, IMAGE_H/8, IMAGE_W/8),
                                                   border_mode=(1, 1))

        self.conv4_4 = VGG19_conv2d_layer(input=self.conv4_3.output,
                                                   layer_name='11',
                                                   image_shape=(None, 512, IMAGE_H/8, IMAGE_W/8),
                                                   border_mode=(1, 1))

        self.pool4 = pool_2d(
            input=self.conv4_4.output,
            ds=(2, 2),
            mode=pool_method,
            ignore_border=True,
            st=(2, 2))

        self.conv5_1 = VGG19_conv2d_layer(input=self.pool4,
                                                   layer_name='12',
                                                   image_shape=(None, 512, IMAGE_H/16, IMAGE_W/16),
                                                   border_mode=(1, 1))
        self.conv5_2 = VGG19_conv2d_layer(input=self.conv5_1.output,
                                                   layer_name='13',
                                                   image_shape=(None, 512, IMAGE_H/16, IMAGE_W/16),
                                                   border_mode=(1, 1))
        self.conv5_3 = VGG19_conv2d_layer(input=self.conv5_2.output,
                                                   layer_name='14',
                                                   image_shape=(None, 512, IMAGE_H/16, IMAGE_W/16),
                                                   border_mode=(1, 1))

        self.conv5_4 = VGG19_conv2d_layer(input=self.conv5_3.output,
                                                   layer_name='15',
                                                   image_shape=(None, 512, IMAGE_H/16, IMAGE_W/16),
                                                   border_mode=(1, 1))

        self.pool5 = pool_2d(
            input=self.conv5_4.output,
            ds=(2, 2),
            mode=pool_method,
            ignore_border=True,
            st=(2, 2))

        # self.dense_1_input= self.pool5.flatten(2)
        #
        # self.dense_1 = DropoutHiddenLayer(is_train = numpy.cast['int32'](0),
        #                    input = self.dense_1_input,
        #                    W=layer_weights['dense_1_weights'],
        #                    b=layer_weights['dense_1_bias'],)
        #
        # self.dense_2 = DropoutHiddenLayer(is_train=numpy.cast['int32'](0),
        #                                input=self.dense_1.output,
        #                                W=layer_weights['dense_2_weights'],
        #                                b=layer_weights['dense_2_bias'], )
        #
        # self.dense_3 = DropoutHiddenLayer(is_train=numpy.cast['int32'](0),
        #                                input=self.dense_2.output,
        #                                W=layer_weights['dense_3_weights'],
        #                                b=layer_weights['dense_3_bias'],
        #                                    activation=softmax )

# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1000, activation='softmax'))


class VGG16(object):

    def __init__(self,input_image_shape,pool_method = 'max'):
        self.input = theano.tensor.tensor4('input')
        self.conv1_1 = VGG19_conv2d_layer(input=self.input,
                                    layer_name='convolution2d_1',
                                    image_shape=input_image_shape,
                                    border_mode=(1,1))

        self.conv1_2 = VGG19_conv2d_layer(input=self.conv1_1.output,
                                              layer_name='convolution2d_2',
                                              image_shape=(None, 64, 224, 224),
                                              border_mode=(1,1))
        self.pool1 = pool_2d(
            input = self.conv1_2.output,
            ds = (2,2),
            mode = pool_method,
            ignore_border=True,
            st = (2,2))

        self.conv2_1 = VGG19_conv2d_layer(input=self.pool1,
                                                   layer_name='convolution2d_3',
                                                   image_shape=(None, 64, 112, 112),
                                                   border_mode=(1,1))

        self.conv2_2 = VGG19_conv2d_layer(input=self.conv2_1.output,
                                                   layer_name='convolution2d_4',
                                                   image_shape=(None, 128, 112, 112),
                                                   border_mode=(1,1))

        self.pool2 = pool_2d(
            input=self.conv2_2.output,
            ds=(2, 2),
            mode=pool_method,
            ignore_border=True,
            st=(2, 2))

        self.conv3_1 = VGG19_conv2d_layer(input=self.pool2,
                                                   layer_name='convolution2d_5',
                                                   image_shape=(None, 128, 56, 56),
                                                   border_mode=(1, 1))
        self.conv3_2= VGG19_conv2d_layer(input=self.conv3_1.output,
                                                   layer_name='convolution2d_6',
                                                   image_shape=(None, 128, 56, 56),
                                                   border_mode=(1, 1))
        self.conv3_3 = VGG19_conv2d_layer(input=self.conv3_2.output,
                                                   layer_name='convolution2d_7',
                                                   image_shape=(None, 128, 56, 56),
                                                   border_mode=(1, 1))

        self.conv3_4 = VGG19_conv2d_layer(input=self.conv3_3.output,
                                                   layer_name='convolution2d_8',
                                                   image_shape=(None, 128, 56, 56),
                                                   border_mode=(1, 1))

        self.pool3 = pool_2d(
            input=self.conv3_4.output,
            ds=(2, 2),
            mode=pool_method,
            ignore_border=True,
            st=(2, 2))

        self.conv4_1 = VGG19_conv2d_layer(input=self.pool3,
                                                   layer_name='convolution2d_9',
                                                   image_shape=(None, 512, 28, 28),
                                                   border_mode=(1, 1))
        self.conv4_2 = VGG19_conv2d_layer(input=self.conv4_1.output,
                                                   layer_name='convolution2d_10',
                                                   image_shape=(None, 512, 28, 28),
                                                   border_mode=(1, 1))
        self.conv4_3 = VGG19_conv2d_layer(input=self.conv4_2.output,
                                                   layer_name='convolution2d_11',
                                                   image_shape=(None, 512, 28, 28),
                                                   border_mode=(1, 1))

        self.conv4_4 = VGG19_conv2d_layer(input=self.conv4_3.output,
                                                   layer_name='convolution2d_12',
                                                   image_shape=(None, 512, 28, 28),
                                                   border_mode=(1, 1))

        self.pool4 = pool_2d(
            input=self.conv4_4.output,
            ds=(2, 2),
            mode=pool_method,
            ignore_border=True,
            st=(2, 2))

        self.conv5_1 = VGG19_conv2d_layer(input=self.pool4,
                                                   layer_name='convolution2d_13',
                                                   image_shape=(None, 512, 14, 14),
                                                   border_mode=(1, 1))
        self.conv5_2 = VGG19_conv2d_layer(input=self.conv5_1.output,
                                                   layer_name='convolution2d_14',
                                                   image_shape=(None, 512, 14, 14),
                                                   border_mode=(1, 1))
        self.conv5_3 = VGG19_conv2d_layer(input=self.conv5_2.output,
                                                   layer_name='convolution2d_15',
                                                   image_shape=(None, 512, 14, 14),
                                                   border_mode=(1, 1))

        self.conv5_4 = VGG19_conv2d_layer(input=self.conv5_3.output,
                                                   layer_name='convolution2d_16',
                                                   image_shape=(None, 512, 14, 14),
                                                   border_mode=(1, 1))
