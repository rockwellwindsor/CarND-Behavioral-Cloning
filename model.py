# Imports
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import json
import random
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from time import time
from pathlib import PurePosixPath
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D
from keras.layers import Convolution2D
from keras.layers import Cropping2D
from keras.optimizers import Adam
from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
# print(Imports imported)

# Variables needed
cameras = ['left', 'center', 'right']
camera_center = ['center']
steering_adj = {'left': 0.25, 'center': 0., 'right': -.25}
time = time() # For logging data timestamps
filter_straights = False
dropout = .40
validation_log_path = ''
validation_size = 2000
epochs = 10
batch_size = 500 
training_size = 50000 # Has to be divided equally by batch_size or you get an error
cnn_model = 'nvidia'
# print('Variables loaded')

# Processing

# This method loads an image and converts it to RGB
def load_image(log_path, filename):
    
    filename = filename.strip()
    if filename.startswith('IMG'):
        filename = log_path+'/'+filename
    elif filename.startswith('/User'): # To load path to own data collected
        filename = filename
    else:
        # load it relative to where log file is now, not whats in it
        filename = log_path+'/IMG/'+PurePosixPath(filename).name

    img = cv2.imread(filename)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# this method randomily changes an image's brightness
# brightness - referenced Vivek Yadav post
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
def randomise_image_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    bv = .25 + np.random.uniform()
    hsv[::2] = hsv[::2]*bv

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# This method crops the camera image to fit nvidia model input shape
def crop_camera(img, crop_height=66, crop_width=200):
    height = img.shape[0]
    width = img.shape[1]
    y_start = 60
    x_start = int(width/2)-int(crop_width/2)

    return img[y_start:y_start+crop_height, x_start:x_start+crop_width]


# This method removes extra rows when driving in a straight line 
def filter_driving_straight(data_df, hist_items=5):

    steering_history = deque([])
    drop_rows = []

    for idx, row in data_df.iterrows():
        # controls = [getattr(row, control) for control in vehicle_controls]
        steering = getattr(row, 'steering')

        # record the recent steering history
        steering_history.append(steering)
        if len(steering_history) > hist_items:
            steering_history.popleft()

        # if just driving in a straight
        if steering_history.count(0.0) == hist_items:
            drop_rows.append(idx)

    # return the dataframe minus straight lines that met criteria
    return data_df.drop(data_df.index[drop_rows])

# referenced Vivek Yadav post
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
def jitter_image_rotation(image, steering):
    rows, cols, _ = image.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange/2
    steering = steering + transX/transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    image = cv2.warpAffine(image, transMat, (cols, rows))
    return image, steering
    
# this method jitters with a random camera image, adjusts steering and randomizes brightness
def jitter_camera_image(row, log_path, cameras):
    steering = getattr(row, 'steering')

    # use one of the cameras randomily
    camera = cameras[random.randint(0, len(cameras)-1)]
    steering += steering_adj[camera]

    image = load_image(log_path, getattr(row, camera))
    image, steering = jitter_image_rotation(image, steering)
    image = randomise_image_brightness(image)

    return image, steering

# http: // machinelearningmastery.com / display - deep - learning - model - training - history - in -keras /
#  This method is not working at the moment, but I will be working on it in the furture so I left it along with the two after it.
def draw_history_graph(history) :
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# Working on creating predictions, this is currently broken
def t_predict(features, labels,cameras):

    log_file = '/driving_log.csv'
    log_path = './data'
    skiprows = 1
    # load the csv log file
    print("Camera: ", cameras)
    print("Log path: ", log_path)
    print("Log file: ", log_file)

    column_names = ['center', 'left', 'right','steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv(log_path+'/'+log_file,names=column_names, skiprows=skiprows)
    data_count = len(data_df)

    # initialise data extract
    t_features = []
    t_labels = []

    row = data_df.iloc[np.random.randint(data_count-1)]
    steering = getattr(row, 'steering')
    image = load_image(log_path, getattr(row, cameras[0]))

    t_features.append(image)
    t_labels.append(steering)

    clf = GaussianNB()
    clf.fit(features, labels)
    pred = clf.pred(t_features)
    print(accuracy_score(pred, t_labels))

# Working on creating predictions, this is currently not working.
def v_predict(features, labels,cameras):

    log_file = '/test_driving_log.csv'
    log_path = './data/test'
    skiprows = 1
    # load the csv log file
    print("Camera: ", cameras)
    print("Log path: ", log_path)
    print("Log file: ", log_file)

    column_names = ['center', 'left', 'right','steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv(log_path+'/'+log_file,names=column_names, skiprows=skiprows)
    data_count = len(data_df)

    # initialise data extract
    t_features = []
    t_labels = []

    i = 0

    while i < 1000:

        row = data_df.iloc[np.random.randint(data_count-1)]
        steering = getattr(row, 'steering')
        image = load_image(log_path, getattr(row, cameras[0]))

        t_features.append(image)
        t_labels.append(steering)

        i += 1

    clf = GaussianNB()
    clf.fit(features, labels)
    pred = clf.pred(t_features)
    print(accuracy_score(pred, t_labels))

# Generating
def gen_train_data(skiprows=1,cameras=cameras, filter_straights=True,crop_image=True, batch_size=128):

    integer = 1
    log_file = '/driving_log.csv'
    log_path = './data'
    # load the csv log file
    print("Cameras: ", cameras)
    print("Log path: ", log_path)
    print("Log file: ", log_file)

    column_names = ['center', 'left', 'right','steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv(log_path+'/'+log_file,names=column_names, skiprows=skiprows)

    # filter out straight line stretches
    # TODAY
    if filter_straights:
        data_df = filter_driving_straight(data_df)

    data_count = len(data_df)

    print("Log with %d rows." % (len(data_df)))

    # Open a log to track this data.
    f = open('data/output_logs/output'+ str(time) +'.txt', 'w')

    while True:  # need to keep generating data

        # Tracking data, prinitng to a text file.
        if integer == 10000:
            print >> f, '#########################################'
            print >> f, features
            print >> f, labels
            print >> f, 'Closing'
            f.close()

        # initialise data extract
        features = []
        labels = []

        # create a random batch to return
        while len(features) < batch_size:
            row = data_df.iloc[np.random.randint(data_count-1)]

            image, steering = jitter_camera_image(row, log_path, cameras)

            steering_check = "%.2f" % round(steering,2)
            
            # Printing this data to the file.
            print >> f, 'Steering check :', steering_check

            # Printing this data to the file.
            if integer < 9999:
                print >> f, 'Row '+ str(integer) +' Original Angle : ' + str(steering) + '' 

            # flip 50% randomily that are not driving straight
            if random.random() >= .5 and abs(steering) > 0.1:
                old_steering = steering
                image = cv2.flip(image, 1)
                steering = -steering

                # Printing this data to the file.
                if integer < 9999:
                    print >> f, 'Row '+ str(integer) +' : '+ str(old_steering) + ' Flipped to ', steering  

            # Angle Random straight line driving.

            elif random.random() >= .8 and steering_check == -0.00:
                chance = random.random()
                if chance % 2:
                    steering = steering - .02
                else:
                    steering = steering + .02

                # Printing this data to the file.
                if integer < 9999:
                    print >> f, 'Row '+ str(integer) +' : '+ str(old_steering) + ' Angled to ', steering  

            # Angle Random straight line driving.
            elif random.random() >= .8 and steering_check == 0.00:
                chance = random.random()
                if chance % 2:
                    steering = steering - .02
                else:
                    steering = steering + .02

                # Printing this data to the file.
                if integer < 9999:
                    print >> f, 'Row '+ str(integer) +' : '+ str(old_steering) + ' Angled to ', steering  

            else:

                # Printing this data to the file.
                print >> f, 'Nothing happened '
                steeering = steering

            if crop_image:
                image = crop_camera(image)

            features.append(image)
            labels.append(steering)

            # Printing this data to the file.
            if integer < 9999:
                print >> f, 'Row '+ str(integer) +' :', steering  # or f.write('...\n')
            integer += 1


        # if integer < 1000:
        #     t_predict(np.array(features), np.array(labels),cameras)

        # yield the batch
        yield (np.array(features), np.array(labels))


# create a valdiation data generator for keras fit_model
def gen_val_data(camera=camera_center[0],crop_image=True, skiprows=1,batch_size=128):

    log_file = '/validation_driving_log.csv'
    log_path = './data'

    # load the csv log file
    print("Camera: ", camera)
    print("Log path: ", log_path)
    print("Log file: ", log_file)

    column_names = ['center', 'left', 'right','steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv(log_path+'/'+log_file,names=column_names, skiprows=skiprows)
    data_count = len(data_df)

    print("Log with %d rows." % (data_count))

    while True:  # need to keep generating data

        # initialize data extract
        features = []
        labels = []

        # create a random batch to return
        while len(features) < batch_size:
            row = data_df.iloc[np.random.randint(data_count-1)]
            steering = getattr(row, 'steering')

            # adjust steering if not center
            steering += steering_adj[camera]

            image = load_image(log_path, getattr(row, camera))

            if crop_image:
                image = crop_camera(image)

            features.append(image)
            labels.append(steering)

        # yield the batch

        yield (np.array(features), np.array(labels))



# Nvidia Model
def build_nvidia_model(img_height=66, img_width=200, img_channels=3,dropout=.4):

    # build sequential model
    model = Sequential()

    # normalisation layer
    img_shape = (img_height, img_width, img_channels)
    model.add(Lambda(lambda x: x * 1./127.5 - 1, input_shape=(img_shape), output_shape=(img_shape), name='Normalization'))

    # convolution layers with dropout
    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    same, valid = ('same', 'valid')
    padding = [valid, valid, valid, valid, valid]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]

    for l in range(len(nb_filters)):
        model.add(Convolution2D(nb_filters[l], kernel_size[l][0], kernel_size[l][1], border_mode=padding[l],subsample=strides[l],activation='elu'))
        model.add(Dropout(dropout))

    # flatten layer
    model.add(Flatten())

    # fully connected layers with dropout
    neurons = [100, 50, 10]
    for l in range(len(neurons)):
        model.add(Dense(neurons[l], activation='elu'))
        model.add(Dropout(dropout))

    # logit output - steering angle
    model.add(Dense(1, activation='elu', name='Out'))

    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer,loss='mse')
    return model


def get_callbacks():

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0,patience=1, verbose=1, mode='auto')
    return [earlystopping]


def main(_):

    crop_image = False

    crop_image = True

    # build model and display layers
    model = build_nvidia_model(dropout=dropout)


    # for l in model.layers:
    #     print(l.name, l.input_shape, l.output_shape,
    #           l.activation if hasattr(l, 'activation') else 'none')
    print(model.summary())

    plot(model, to_file='model.png', show_shapes=True)

    model.fit_generator(
        gen_train_data(
            cameras=cameras,
            crop_image=crop_image,
            batch_size=batch_size
        ),
        samples_per_epoch=training_size,
        nb_epoch=epochs,
        callbacks=get_callbacks(),
        validation_data=gen_val_data(
            crop_image=crop_image,
            batch_size=batch_size),
            nb_val_samples=validation_size
        )

    # # Show history
    # history = h.history
    # draw_history_graph(history)

    # save weights and model
    model.save_weights('model.h5')
    with open('model.json', 'w') as modelfile:
        json.dump(model.to_json(), modelfile)


# calls the `main` function above
if __name__ == '__main__':
    tf.app.run()