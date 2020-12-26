from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import numpy as np
import pandas as pd

import cv2                  
import os 

from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model

def getPrediction(filename):

    # load the labels
    r = np.load("XandZ.npz") 
    Z = r["arr_1"] 
    ohe = OneHotEncoder()
    ohe.fit(Z.reshape(-1,1))
    label_array = ohe.categories_[0]

    DIR = 'uploads'  # upload the image into this folder
    img_path = os.listdir(DIR)[0] 
    IMG_SIZE = 64

    path = os.path.join(DIR,img_path)
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))  # resize the image
    # cv2_imshow(img)  # show the resized image
    # print('Resized Dimensions : ',img.shape)

    X=[]
    X.append(np.array(img))

    # normalized the data
    X = np.array(X)  # convert X from list to np.array
    X = X.astype('float32')/255  # normalized the data to 0-1

    model = load_model('vgg16_conv_tuning_last2conv_modelsave.h5')
    model.load_weights('vgg16_vege_conv_tuning_2.hdf5')
    Y = model.predict(X)
    predicted_label = label_array[np.argmax(Y)]

    # model = VGG16()
    # image = load_img('uploads/'+filename, target_size=(224, 224))
    # image = img_to_array(image)
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # image = preprocess_input(image)
    # yhat = model.predict(image)
    # label = decode_predictions(yhat)
    # label = label[0][0]
    # print('%s (%.2f%%)' % (label[1], label[2]*100))
    # return label[1], label[2]*100
    return predicted_label
