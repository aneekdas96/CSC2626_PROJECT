import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, Subtract, Activation, BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K

def siamese_net():
    
    input_shape = (240,240,3)

    left_input = Input(input_shape)
    right_input = Input(input_shape)

    model = keras.models.Sequential([
        Conv2D(96,(11,11),strides = 4,name="conv0"),
        BatchNormalization(axis = 3 , name = "bn0"),
        Activation('relu'),
        MaxPooling2D((3,3),strides = 2,name = 'max0'),
        
        Conv2D(256,(5,5),padding = 'same' , name = 'conv1'),
        BatchNormalization(axis = 3 ,name='bn1'),
        Activation('relu'),
        MaxPooling2D((3,3),strides = 2,name = 'max1'),
        
        Conv2D(384, (3,3) , padding = 'same' , name='conv2'),
        BatchNormalization(axis = 3, name = 'bn2'),
        Activation('relu'),
        
        Conv2D(384, (3,3) , padding = 'same' , name='conv3'),
        BatchNormalization(axis = 3, name = 'bn3'),
        Activation('relu'),
        
        Conv2D(256, (3,3) , padding = 'same' , name='conv4'),
        BatchNormalization(axis = 3, name = 'bn4'),
        Activation('relu'),

        Conv2D(200, (3,3) , padding = 'same' , name='conv5'),
        BatchNormalization(axis = 3, name = 'bn5'),
        Activation('relu'),

        Flatten(),
        Dense(200, activation = 'relu', name = 'fc0'),
        Dense(200, activation = 'relu', name = 'fc1')
    ])

    encoded_l = model(left_input)
    encoded_r = model(right_input)

    subtracted = Subtract()([encoded_l, encoded_r])
    prediction = Dense(4, activation='sigmoid')(subtracted)
    siamese_net = Model(input=[left_input, right_input], output=prediction)
    
    return siamese_net
