# Get the model to take in the images.
# Output the steering angle.
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
import os
import theano
from PIL import Image
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

print('Imports imported')

# Image dimensions and number of channels.  3 = color, 1 = grayscale
img_rows, img_cols, img_channels = 160, 320, 3

path1 = './data/IMG/'          #path of folder of images    
path2 = './data/PROCESSED_IMG'  #path of folder to save images  
path3 = './data/AUGMENTED_IMG'

CROPPING = (54, 0, 0, 0)
SHIFT_OFFSET = 0.2
SHIFT_RANGE = 0.2

print('PATHS loaded')

# Visualize Data

#reading in an image
image = mpimg.imread('./data/IMG/center_2016_12_01_13_30_48_287.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)

listing = os.listdir(path1) 
num_samples=size(listing)
print('FILES loaded')

images = []
aug_images = []
int = 0

# randomily change the image brightness
def randomise_image_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # brightness - referenced Vivek Yadav post
    # https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0

    bv = .25 + np.random.uniform()
    hsv[::2] = hsv[::2]*bv

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

for file in listing:
    im = Image.open(path1 + "/" + file)            
    img.save(path2 + '/' +  file, "JPEG")
    img = np.asarray(img, dtype='float32')
    images.append(img)

    if int % 5 == 0:
        aug_img = randomise_image_brightness(img)
        aug_images.append(aug_img)
        int += 1

    int += 1

images = np.array(images, dtype='float32')
print(len(images))
print(len(aug_images))
print('UPDATED FILES saved')



