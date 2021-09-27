import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

def segnet(pretrained_weights = None,input_size = (512,512,1)):
    #encoder
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    #(128,128)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv4 = Conv2D(128, 3,  padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    #(64,64)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv6 = Conv2D(256, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv7 = Conv2D(256, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)


    #(32,32) 
    conv8 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv9 = Conv2D(512, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv10 = Conv2D(512, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv9)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv10)
    
    #(16,16) 
    conv11 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv12 = Conv2D(512, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv11)
    conv13 = Conv2D(512, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv12)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv13)
    
    #(8,8)  
    #decoder  
    UP1 = UpSampling2D(size = (2,2),interpolation='bilinear')(pool5)
 
    #(16,16)  
    conv14 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UP1)
    conv15 = Conv2D(512, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv14)
    conv16 = Conv2D(512, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv15)
    UP2 = UpSampling2D(size = (2,2),interpolation='bilinear')(conv16)
    ADD1 = concatenate([conv10,UP2], axis = 3)
   
    #(32,32) 
    conv17 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ADD1)
    conv18 = Conv2D(512, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv17)
    conv19 = Conv2D(512, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv18)
    
    UP3 = UpSampling2D(size = (2,2),interpolation='bilinear')(conv19)
    ADD2 = concatenate([conv7,UP3], axis = 3)
    
    #(64,64)  
    conv20 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ADD2)
    conv21 = Conv2D(256, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv20)
    conv22 = Conv2D(256, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv21)
    UP4 = UpSampling2D(size = (2,2),interpolation='bilinear')(conv22)
    ADD3 = concatenate([conv4,UP4], axis = 3)
    
    #(128,128)
    conv23 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ADD3)
    conv24 = Conv2D(128, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv23)
    conv25 = Conv2D(128, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv24)
    UP5 = UpSampling2D(size = (2,2),interpolation='bilinear')(conv25)
    ADD4 = concatenate([conv2,UP5], axis = 3)
    
    #(256,256) 
    conv26 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(ADD4)
    conv27 = Conv2D(64, 3, activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv26)
    conv28 = Conv2D(1, 1,  padding = 'same', kernel_initializer = 'he_normal')(conv27)
    AC28 = Activation('sigmoid')(conv28)
    
    model = Model(input = inputs, output = AC28)
    model.compile(optimizer = Adam(lr = 5e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])    
        
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model
