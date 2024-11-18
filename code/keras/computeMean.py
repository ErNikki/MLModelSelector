import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.applications
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from classification_models.keras import Classifiers
from keras.optimizers import SGD
#'.mdl_wts.hdf5'
import os
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

""" DATASET FOLDER """
PATH_DATASET_FOLDER = os.getcwd() + "/../../dataset/SUN397"

""" MODELS SUB FOLDERS """
PATH_TRAIN_MODELS_FOLDER=PATH_DATASET_FOLDER+"/train_models"
PATH_VAL_MODELS_FOLDER=PATH_DATASET_FOLDER+"/val_models"

""" TEST PATH FOLDERS """
PATH_TEST_FOLDER=PATH_DATASET_FOLDER+"/test"

""" PATH TO SAVE MODEL """
PATH_TO_SAVE_MODELS="./models"

train_ds=tf.keras.preprocessing.image_dataset_from_directory(
        PATH_TRAIN_MODELS_FOLDER,
        labels='inferred',
        label_mode='categorical',
        #class_names=None,
        color_mode='rgb',
        batch_size=32,
        image_size=(255,255),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False,
        #pad_to_aspect_ratio=False,
        #data_format=None,
        #verbose=True
    )

def mean(x):
    mean_, variance_=tf.nn.moments(x,axes=[0,1,2])
    return mean_, variance_

mean = 0.
std = 0.
nb_samples = 0.

for imgs,labels in train_ds:
    #batch_samples = data.size(0)
    #data = data.view(batch_samples, data.size(1), -1)
    #mean += data.mean(2).sum(0)
    #std += data.std(2).sum(0)
    #nb_samples += batch_samples
    mean+=tf.math.reduce_mean(imgs, axis=None, keepdims=False, name=None)
    std+=tf.math.reduce_std(imgs, axis=None, keepdims=False, name=None)

    nb_samples+=1

mean /= nb_samples*32
std /= nb_samples*32

print(mean)
print(std)