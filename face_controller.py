# This model will contain the code to call the functions in the face_transformer module
# To make a fake face.

import pandas as pd 
import face_transformer as ftf
import boto3 # to connect to my asws s3 image repository (> 200000 images)
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import cv2

# These three lines are for suppressing warning messages.
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.experimental
import keras.applications.mobilenet
from keras.preprocessing import image 


# Some notes here
# I created three models - one for glasses/no glasses, one for male/female, and one
# for facial hair (beard, goatee, mustache).  
# The models return five numbers, of which the last three are too small to be
# meaningful. 
# For glasses/no glasses the first one indicates glasses, the second no glasses.
# For gender, the first one indicates male, the second female.
# For facial hair, the first one indicates hair, the second no hair.  This is not a good 
# model though. But I will use it and see what happens.

# Prepares an image - the models need the images to be 224 x 224

FACE1 = None
FACE2 = None
image_folder = './temp'
gen_model = tensorflow.keras.models.load_model('./models/gender_model_1.h5',custom_objects={'KerasLayer':hub.KerasLayer})
glas_model = tensorflow.keras.models.load_model('./models/glasses_model_1.h5',custom_objects={'KerasLayer':hub.KerasLayer})
hair_model = tensorflow.keras.models.load_model('./models/facehair_model_1.h5',custom_objects={'KerasLayer':hub.KerasLayer})
  

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def clear_temp_folder():    
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            path = os.path.join(image_folder, file)
            os.remove(path)

def get_random_filename():
    base = "00000"    
    num1 = str(random.randrange(1,200000))
    pad = base[:(6 - len(num1))]    
    file_num = pad + num1 + '.jpg'   
    return file_num

# retrieves 20 random images to work with
def get_random_images():
    s3client = boto3.client('s3')
    img_list = list()
    for i in range(20):
        img_list.append(get_random_filename())            
    for item in img_list:
        s3client.download_file('fakefacemaker',item, os.path.join(image_folder, item))
    

def check_for_male(image_file):    
    result = gen_model.predict(image_file)
    if result[0][0] > result[0][1]:
        return True
    else:    
        return False
    
def check_for_glasses(image_file):    
    result = glas_model.predict(image_file)
    if result[0][0] > result[0][1]:
        return True
    else:    
        return False    

def check_for_hair(image_file):    
    result = hair_model.predict(image_file)
    if result[0][0] > result[0][1]:
        return True
    else:    
        return False
           

# Goes out and gets two reasonably compatible images from my aws s3 repository.
def retrieve_two_images():
    image1 = None
    image2 = None
    f1 = None
    f2 = None
    files_found = 0
    found_file_gender = 0  # 1 = male, 2 = female  
    # Get one image from the image folder
    for root, dirs, files in os.walk(image_folder, topdown=False):
        for f in files: 
            if files_found > 1:
                break 
            im = prepare_image(os.path.join(image_folder,f))
            if not check_for_glasses(im):                 
            
                if files_found == 0:
                    if check_for_male(im):
                        if not check_for_hair(im):                        
                            image1 = im
                            files_found += 1
                            found_file_gender = 1
                            f1 = os.path.join(image_folder,f)                            
                    else:
                        image1 = im
                        files_found += 1
                        found_file_gender = 2
                        f1 = os.path.join(image_folder,f)                                     
                        
                elif files_found == 1:
                    if check_for_male(im):
                        if not check_for_hair(im):                       
                            if found_file_gender == 1:
                                image2 = im
                                files_found += 1
                                f2 = os.path.join(image_folder,f)
                                   
                    else:
                        if found_file_gender == 2:
                            image2 = im
                            files_found += 1
                            f2 = os.path.join(image_folder,f)
                               
    return f1, f2 

def grab_images():
    clear_temp_folder()
    get_random_images()
    global FACE1, FACE2
    FACE1, FACE2 = retrieve_two_images()
        
    
# This function runs all the others in order to generate an image with a fake face.
def fake_face_generator():    
    
    good_images = False
    grab_images()
    while good_images == False:        
        try: 
            # These two functions are from Matthew Earl's github account.
            # Mr. Earl uses a python dlib function called a detector to identify a face
            # within an image, which allows the generation of "landmarks."
            # It is possible for the detector to fail to find a face in one of 
            # the images.  So, in case it fails:            
            image1, landmarks1 = ftf.read_im_and_landmarks(FACE1)
            image2, landmarks2 = ftf.read_im_and_landmarks(FACE2)
            good_images = True
        except:
            grab_images()    

    # From here on is Matthew Earl's code 
    tr_landmarks2 = ftf.transformation_from_points(landmarks1[ftf.ALIGN_POINTS], landmarks2[ftf.ALIGN_POINTS])
        
    mask2 = ftf.get_face_mask(image2, landmarks2)
    warped_mask = ftf.warp_im(mask2, tr_landmarks2, image1.shape)
    combined_mask = np.max([ftf.get_face_mask(image1, landmarks1), warped_mask], axis=0)
    
    warped_image2 = ftf.warp_im(image2,tr_landmarks2, image1.shape)
    warped_corrected_im2 = ftf.correct_colours(image1, warped_image2, landmarks1)   

    output_im = image1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask

    # Another step is needed here (missing from Matthew Earl's code) - which is to 
    # convert the output_im from floats into ints.
    result = output_im.astype(int)
    
    return image1, image2, result


# So that we can run this from the command line.
if __name__ == '__main__': 
    im1, im2, res = fake_face_generator()   
    plt.imshow( im1 )
    plt.show()
    plt.imshow( im2 )
    plt.show()
    plt.imshow( res )
    plt.show()
    
    

    


    




