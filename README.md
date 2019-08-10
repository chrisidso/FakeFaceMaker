Fake Face Maker is a project designed and executed by Christopher Idso to be the final project (capstone) as a student at Galvanize.

Fake Face Maker creates a fake face by extracting facial features from one image and inserting them into another image and displaying the resulting image.  It does this by:

1) Downloading 20 images into a temporary folder.
2)  Selecting two of them (using trained Keras/Tensorflow models) so that the images
     are reasonably compatible (both male, both female, no eyeglasses, no facial hair).
3)  Aligning and rotating the second image to match the first.
4)  Extracting facial features from the second image.
5)  Inserting them into the first image.
6)  Displaying the resulting image.

Items 1,2 and 6 above were accomplished using original code (in the python module called:
face_controller.py).  It uses three models which are stored locally in the “models” folder.

Items 3,4 and 5 were accomplished using code borrowed from Matthew Earl (stored locally in the python module called:  face_transformer.py) – and which resides on his github account:

https://matthewearl.github.io/2015/07/28/switching-eds-with-python/

Please note that he has a copyright on his code, and that his copyright notice has been included in the module containing his code (face_transformer.py).

He walks the reader through what his code does and adds a link where you can download the complete code.   There is a predictor model he uses which he provides a handy link to and which is stored locally in the ‘predictor’ folder. 

Fake Face Maker can be run by down loading the two python modules, and the models, predictor, and temp folders, and the jupyter notebook called “fake_face_maker_run_one.ipynb”  - all into the same directory.  

Then open and run the notebook, or at the command the line (and from within the folder the files were downloaded to) type:       python face_controller.py

In order for the code contained in this project to run I had to download and install  a a number of python libraries onto my laptop (a linux box running Ubuntu).   

The following jupyter notebooks:

train_facehair
train_gender
train_glasses

 are what I used to train my models. 
The following notebooks:

facehair_image_processor
gen_image_processor
glass_image_processor

 are what I used to create the folders that contain the train/validation images, which are then used by the other three notebooks (above) to train the models.    None of these
notebooks will run for you, but feel free to look at them.  

My respository of images is at Amazon Web Services in an s3 bucket called fakefacemaker. They are also online at this link:

http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html


This project proved to be more difficult than I thought it would be.   Along the way I encountered the following problems: 

The model I chose required its images to be of a specific size which was different than the size of the images I was using.  So I had to work out an efficient way to change the size of my images.

I knew that I would have to be able to save my models once I trained them, so that the code I intended to write later would be able to load the models from a folder.   Saving my models was not a problem, but loading them back in was difficult because the usual ways to do that (model object’s load method, python’s pickle, python’s dill library) did not work.  I finally found a way to load my models back in.   

Now (finally) I could train and save the three models I needed.  In testing the facial hair model I found that it did not work very well.  Facial hair is difficult to detect, it seems.
The other two models worked nicely.

To begin the process of making the fake face I wanted to download images one at a time and examine them.  But I could not figure out how to do that, so settled for downloading a bunch of image files to a folder, where I could apply some logic to choose two of them.

Now I needed to get Mr. Earl’s code to work to complete the process, and I found that his code was incomplete.  But I got it to work in the end.


