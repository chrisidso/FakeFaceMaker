import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.transform
import skimage.filters
import skimage.feature
import skimage.restoration
from skimage.transform import resize
from skimage.filters import sobel
from skimage.feature import canny
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle
from skimage.io import imread, imshow
from sklearn.cluster import KMeans
from scipy import misc
from skimage.color import rgb2gray
from skimage.io import imread, imshow
import shutil
import glob

import boto3
import sys
import os
import pickle

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
# tf.enable_eager_execution()

import tensorflow_hub as hub
# import tensorflow_datasets as tfds

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.experimental
import keras.applications.mobilenet
from keras.preprocessing import image

# Resizes an image
def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

# a function for resizing images in a target directory
def image_mover(target_dir, working_dir):
    # copy files from target to working
    for root, dirs, files in os.walk(target_dir, topdown=False):
        for f in files:
            shutil.move(os.path.join(target_dir,f), os.path.join(working_dir,f))


# empty target
def image_resizer(target_dir, working_dir, width, height):   
    # for each file in working
    for root, dirs, files in os.walk(working_dir, topdown=False):
        for f in files:
            im1 = Image.open(os.path.join(working_dir,f))
            im2 = im1.resize( (width, height) )
            im2.save(os.path.join(target_dir, f))      


def getImData(data):
    image = Image.open(data)
    return image.size


def getImageInfo(data):
    data = str(data)
    size = len(data)
    height = -1
    width = -1
    content_type = ''    
    
    content_type = 'image/jpeg'
    jpeg = io.StringIO(data)
    jpeg.read(2)
    b = jpeg.read(1)
    try:
        while (b and ord(b) != 0xDA):
            while (ord(b) != 0xFF): b = jpeg.read(1)
            while (ord(b) == 0xFF): b = jpeg.read(1)
            if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                jpeg.read(3)
                h, w = struct.unpack(">HH", jpeg.read(4))
                break
            else:
                jpeg.read(int(struct.unpack(">H", jpeg.read(2))[0])-2)
            b = jpeg.read(1)
        width = int(w)
        height = int(h)
    except struct.error:
        pass
    except ValueError:
        pass   
    
    return content_type, width, height                      