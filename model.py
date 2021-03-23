import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, Subtract, Activation, BatchNormalization, Concatenate
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K

def conv_net(right):

    input_shape = (240,240,3)

    im_input = Input(input_shape)

    x = Conv2D(96,(11,11),strides = 4)(im_input)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3),strides = 2)(x)
        
    x = Conv2D(256,(5,5))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3,3),strides = 2)(x)
        
    x = Conv2D(384, (3,3))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
        
    x = Conv2D(384, (3,3))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
        
    x = Conv2D(256, (3,3))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)

    x = Conv2D(200, (3,3))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dense(200, activation = 'relu')(x)
    output = Dense(200, activation = 'relu')(x)
    if right:
        output = Dense(200, activation = 'relu', trainable = False)(x)

    conv_network = Model(im_input, output)

    return conv_network


def dual_net():
    
    input_shape = (240,240,3)

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    left_net = conv_net(False)
    right_net = conv_net(True)

    encoded_l = left_net(left_input)
    encoded_r = right_net(right_input)

    concatted = Concatenate()([encoded_l, encoded_r])
    prediction = Dense(4, activation='sigmoid')(concatted)
    Dual_net = Model(input=[left_input, right_input], output=prediction)
    
    return Dual_net
