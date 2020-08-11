import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pickle
import numpy as np

import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense
from keras.models import Model





def get_labelled_data(x_path, y_path):

    with open('pickle_files/{}.pickle'.format(x_path),'rb') as f:
        train_x = pickle.load(f)
    
    with open('pickle_files/{}.pickle'.format(y_path),'rb') as f:
        train_y = pickle.load(f)
        
    train_x = np.array(train_x)
    train_x = np.expand_dims(train_x, axis=3)
    train_x = train_x.astype('float32')
    train_x = train_x / 255
    
    return train_x, np.array(train_y), tuple(train_x.shape[1:]) # x, y, input_shape


def get_model(input_shape):

    input_layer  = Input(shape=input_shape, name='frame')
    hidden_layer = Conv2D(filters=8, kernel_size=3, strides=2, padding='same', activation='relu')(input_layer)
    hidden_layer = MaxPooling2D(pool_size=(5, 5),strides=(3, 3))(hidden_layer)
    hidden_layer = Conv2D(filters=16, kernel_size=3, strides=2, padding='same', activation='relu')(hidden_layer)
    hidden_layer = MaxPooling2D(pool_size=(5, 5),strides=(3, 3))(hidden_layer)
    hidden_layer = Flatten()(hidden_layer)
    hidden_layer = Dense(16, activation='relu')(hidden_layer)
    output_layer = Dense(4, activation='sigmoid')(hidden_layer) # sigmoid is used to ensure output falls between 0 and 1

    return Model(input_layer, output_layer, name='shintyan_detector')


def my_loss(true, pred):
    
    loss = K.mean(K.abs(true - pred))
    
    constraint = (K.maximum(pred[:,0] - pred[:,2], 0) + K.maximum(pred[:,1] - pred[:,3], 0)) * 10
    # predicted value consists of 2 coordinates (x1,y1) and (x2,y2) that 
    # enclose a box surrounding the object to be detected. x2 should > x1 and y2 should > y1
  
    return loss + constraint


