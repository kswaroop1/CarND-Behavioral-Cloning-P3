import pandas
import numpy as np
import cv2
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

########################################################
# The model is defined below.  It is a variation of NVIDIA architecture with a couple of
# additional layers for recognising extra features and lane line colour independence.
########################################################
def nvidia():
    model= Sequential([
                     Lambda(lambda x: x/127.5 - 1., input_shape=(66,200,3)),  # Normalisation
                     Convolution2D(24, 5, 5, init='he_normal', subsample=(2,2), activation='elu'),
                     Convolution2D(36, 5, 5, init='he_normal', subsample=(2,2), activation='elu'),
                     Convolution2D(48, 5, 5, init='he_normal', subsample=(2,2), activation='elu'),
                     Convolution2D(64, 3, 3, init='he_normal', subsample=(1,1), activation='elu'),
                     Convolution2D(64, 3, 3, init='he_normal', subsample=(1,1), activation='elu'),

                     # These are two additional convolution layers not in the NVIDIA architecture
                     #    vvvvvvvvvvvvvvvvvvvvvvvvvvv
                     Convolution2D(128, 3, 3, init='he_normal', subsample=(1,1), activation='elu', border_mode='same'),
                     Convolution2D(256, 1, 1, init='he_normal', subsample=(1,1), activation='elu'),
                     Dropout(0.25),
                     #   ^^^^^^^^^^^^^^^^^^^
                     Flatten(),
                     Dense(1164, activation='elu', init='he_normal'),
                     Dense(100, activation='elu', init='he_normal'),
                     Dense(50, activation='elu', init='he_normal'),
                     Dense(10, activation='elu', init='he_normal'),
                     Dense(1, init='he_normal') # Linear Regression
                     ])
    model.compile(optimizer='adam', loss='mean_squared_error') # ADAM optimiser, MSE loss
    show_layer_shapes(model) # Shows input and output shapes for a layer
    return model

# Read Udacity provided training data into memory, store in a dictionary
image_data_cache={}
def read_data(dir):
    if dir in image_data_cache.keys():
        return image_data_cache[dir]
    driving_log = pandas.read_csv(dir+'/driving_log.csv')
    left,center,right,rows=[],[],[],len(driving_log)
    for ind in range(rows):
        if ind%int(rows/10)==0: print('Reding file #', ind, ' of ', rows)
        left.append(mpimg.imread(dir+'/'+driving_log['left'][ind].lstrip()))
        center.append(mpimg.imread(dir+'/'+driving_log['center'][ind].lstrip()))
        right.append(mpimg.imread(dir+'/'+driving_log['right'][ind].lstrip()))
    # apply exponetial weighted smoothing to steering angles
    angles = driving_log['steering']
    angles_array = np.asarray(angles)
    fwd = pandas.ewma(angles_array, span=20)
    bwd = pandas.ewma(angles_array[::-1], span=20)
    smooth = np.vstack((fwd, bwd[::-1]))
    smooth = np.mean(smooth, axis=0)
    angles = np.ndarray.tolist(smooth)
    image_data_cache[dir]=(left,center,right,driving_log,angles)
    return image_data_cache[dir]

def show_layer_shapes(model, visual=False):
    if visual:
        SVG(model_to_dot(simplenet).create(prog='dot', format='svg'))
        return
    for i in range(len(model.layers)):
        lay=model.layers[i]
        print('#',i,':',lay.input_shape, '==>',lay.output_shape) #, ', using ', lay.activation)

callbacks = [EarlyStopping(monitor='loss', patience=3),
                     ModelCheckpoint('weights.{epoch:02d}-{loss:.4f}.h5', monitor='loss', save_best_only=True)]

# Image Augmentation Code as described in Vivek Yadav's article
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
# Load Data
import pickle
import math
import matplotlib
import pylab as pl
import pandas
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import time

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
def trans_image(image,steer,trans_range):
    # Translation
    cols,rows=image.shape[1],image.shape[0]
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr,steer_ang
def preprocessImage(image, new_size_col, new_size_row):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col, new_size_row), interpolation=cv2.INTER_AREA) 
    return image
def preprocess_image_file(image, y_steer, new_size_col, new_size_row):
    image,y_steer = trans_image(image,y_steer,100)
    image = augment_brightness_camera_images(image)
    image = preprocessImage(image, new_size_col, new_size_row)
    image = np.array(image)
    return image,y_steer

# Image Augmentation: python generator
def generator(left,center,right,steering, batch_size = 32, new_size_col=200, new_size_row=66, straight_steer=0.1, drop_straight_prob=.5):
    #print('len(left,center,right,steering)=({0},{1},{2},{3})'.format(len(left),len(center),len(right),len(steering)))
    batch_images = np.zeros((batch_size, new_size_row, new_size_col,3))
    batch_steering = np.zeros(batch_size)
    while 1:
        i = 0
        while i<batch_size:
            idx = np.random.randint(len(steering))
            image,mult = {0:(left[idx],1), 1:(center[idx],0), 2:(right[idx],-1)}[np.random.randint(3)]
            steering_adjust = 0.25 # for left or right angle camera
            y_steer = steering[idx] + steering_adjust*mult #- math.copysign(driving_log['steering'][i]*0.2, mult)
            keep_pr = 0
            while keep_pr == 0:
                x,y = preprocess_image_file(image, y_steer, new_size_col, new_size_row)
                if abs(y)<straight_steer:
                    pr_val = np.random.uniform()
                    if pr_val>drop_straight_prob:
                        keep_pr = 1
                else:
                    keep_pr = 1
            batch_images[i] = x
            batch_steering[i] = y
            i+=1
            batch_images[i] = cv2.flip(x,1) # flip image
            batch_steering[i] = -y               # and steering angle
            i+=1
        yield batch_images, batch_steering

batch_size=256
model=nvidia()
left,center,right,driving_log,y_train=read_data('data')
y_train=np.array(driving_log['steering']) # don't use smoothened data
nb_samples=math.ceil(3*len(y_train)/batch_size)*batch_size
hist = model.fit_generator(generator(left,center,right,y_train,batch_size, 200, 66, drop_straight_prob=0),
                    samples_per_epoch=nb_samples, nb_epoch=64,
                    #validation_data=generator(leftv,centerv,rightv,y_v,batch_size, 200, 66, drop_straight_prob=0),
                    #nb_val_samples=math.ceil(4000./batch_size)*batch_size,
                    callbacks=callbacks,
                    verbose=1, nb_worker=12, pickle_safe=True)

model.save('model.h5')
print('Training Completed.\n')
