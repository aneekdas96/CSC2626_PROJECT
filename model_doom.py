import tensorflow as tf
import keras
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Activation, BatchNormalization, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Softmax


def dual_net():

    #Architecture
    conv1 = Conv2D(96, (11,11),strides = 4)
    batch1 = BatchNormalization(axis = 3)
    act1 = Activation('relu')
    pool1 = MaxPooling2D((3,3),strides = 2)
        
    conv2 = Conv2D(256, (5,5))
    batch2 = BatchNormalization(axis = 3)
    act2 = Activation('relu')
    pool2 = MaxPooling2D((3,3),strides = 2)
        
    conv3 = Conv2D(384, (3,3))
    batch3 = BatchNormalization(axis = 3)
    act3 = Activation('relu')
        
    conv4 = Conv2D(384, (3,3))
    batch4 = BatchNormalization(axis = 3)
    act4 = Activation('relu')
        
    conv5 = Conv2D(256, (3,3))
    batch5 = BatchNormalization(axis = 3)
    act5 = Activation('relu')

    conv6 = Conv2D(200, (3,3))
    batch6 = BatchNormalization(axis = 3)
    act6 = Activation('relu')

    flatten = Flatten()
    concatted = Concatenate()

    dense1 = Dense(200, activation = 'relu')
    dense2 = Dense(200, activation = 'relu')
    pred = Dense(4, activation = 'softmax')

    # Inputs through Layers
    left_input_shape = (240, 320, 3)
    right_input_shape = (240, 320, 3    )

    left_input = Input(left_input_shape)
    right_input = Input(right_input_shape)
    print('Finished processing inputs')

    left_op1 = conv1(left_input)
    right_op1 = conv1(right_input)

    left_op2 = batch1(left_op1)
    right_op2 = batch1(right_op1)

    left_op3 = act1(left_op2)
    right_op3 = act1(right_op2)

    left_op4 = pool1(left_op3)
    right_op4 = pool1(right_op3)

    left_op5 = conv2(left_op4)
    right_op5 = conv2(right_op4)

    left_op6 = batch2(left_op5)
    right_op6 = batch2(right_op5)

    left_op7 = act2(left_op6)
    right_op7 = act2(right_op6)

    left_op8 = pool2(left_op7)
    right_op8 = pool2(right_op7)

    left_op9 = conv3(left_op8)
    right_op9 = conv3(right_op8)

    left_op10 = batch3(left_op9)
    right_op10 = batch3(right_op9)

    left_op11 = act3(left_op10)
    right_op11 = act3(right_op10)

    left_op12 = conv4(left_op11)
    right_op12 = conv4(right_op11)

    left_op13 = batch4(left_op12)
    right_op13 = batch4(right_op12)

    left_op14 = act4(left_op13)
    right_op14 = act4(right_op13)

    left_op15 = conv5(left_op14)
    right_op15 = conv5(right_op14)

    left_op16 = batch5(left_op15)
    right_op16 = batch5(right_op15)

    left_op17 = act5(left_op16)
    right_op17 = act5(right_op16)

    left_op18 = conv6(left_op17)
    right_op18 = conv6(right_op17)

    left_op19 = batch6(left_op18)
    right_op19 = batch6(right_op18)
    
    left_op20 = act6(left_op19)
    right_op20 = act6(right_op19)

    left_flat = flatten(left_op20)
    right_flat = flatten(right_op20)

    op1 = concatted([left_flat,right_flat])
    op2 = dense1(op1)
    op3 = dense2(op2)
    prediction = pred(op3)

    Dual_net = Model(inputs=[left_input, right_input], outputs=prediction)

    print(Dual_net.summary())
    return Dual_net
