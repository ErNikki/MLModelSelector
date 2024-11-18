import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.applications
from keras.models import Model,load_model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
import keras_cv
from classification_models.keras import Classifiers
from keras.optimizers import SGD
#'.mdl_wts.hdf5'
import os
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split

""" DATASET FOLDER """
PATH_DATASET_FOLDER = os.getcwd() + "/../../dataset/SUN397"

""" MODELS SUB FOLDERS """
PATH_TRAIN_MODELS_FOLDER=PATH_DATASET_FOLDER+"/train_models"
PATH_VAL_MODELS_FOLDER=PATH_DATASET_FOLDER+"/val_models"

""" TEST PATH FOLDERS """
PATH_TEST_FOLDER=PATH_DATASET_FOLDER+"/test"

""" PATH TO SAVE MODEL """
PATH_TO_SAVE_MODELS="./models"

#devices = tf.config.experimental.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(devices[0], True)

def get_wrap_model(name, input_shape, num_classes):
    
    inputs=keras.Input(shape=input_shape)
    
    if name=="MobileNetV3S":
        base_model=keras.applications.MobileNetV3Small(
            input_shape=input_shape,
            alpha=1.0,
            minimalistic=False,
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            classes=1000,
            pooling=None,
            dropout_rate=0.2,
            #classifier_activation="softmax",
            include_preprocessing=True,
        )
        #scale_layer=tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
        
    elif name=="NASNetMobile":
        base_model=keras.applications.NASNetMobile(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            pooling=None,
            classes=1000,
            #classifier_activation="softmax",
        )
        #scale_layer=keras.applications.nasnet.preprocess_input(inputs)
        
    elif name=="ResNet18":
        ResNet18, scale_layer = Classifiers.get('resnet18')
        base_model = ResNet18(input_shape, weights='imagenet',include_top=False)
        inputs=keras.Input(shape=input_shape)
        #scale_layer=keras.applications.resnet.preprocess_input(inputs)
        
    elif name=="VGG16":
        base_model=keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        #input_tensor=None,
        input_shape=input_shape,
        #pooling=None,
        #classes=1000,
        #classifier_activation="softmax",
        )
        #scale_layer=keras.applications.vgg16.preprocess_input(inputs)
    elif name=="EfficientNetB7":
        base_model=keras.applications.EfficientNetB7(
            include_top=False,
            weights="imagenet",
            #input_tensor=None,
            input_shape=input_shape,
            #pooling=None,
            classes=1000,
            #classifier_activation="softmax",
            #**kwargs
        )
    elif name=="InceptionResNetV2":
        base_model=keras.applications.InceptionResNetV2(
            include_top=False,
            weights="imagenet",
            #input_tensor=None,
            input_shape=input_shape,
            #pooling=None,
            classes=1000,
            #classifier_activation="softmax",
        )
    
    #base_model.trainable = False
    # add new classifier layers
    #flat1 = Flatten()(base_model.layers[-1].output)
    #class1 = Dense(1024, activation='relu')(flat1)
    #output = Dense(num_classes, activation='softmax')(class1)
    # add a global spatial average pooling layer
    
    #scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    #x = scale_layer(inputs)

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    """
    to include scale layer in the model
    """
    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)
    #x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)

    #model.summary(show_trainable=True)

    # A Dense classifier with a single unit (binary classification)
    #outputs = keras.layers.Dense(num_classes)(x)
    #model = keras.Model(inputs, outputs)
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    #x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    #predictions = Dense(num_classes, activation='softmax')(x)
    predictions = Dense(num_classes)(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    """
    return model, base_model

def FitAndFinetuning(base_model, model, train_ds, val_ds, path_save_loss, path_save_accuracy, num_workers):
    """
    for layer in base_model.layers:
        layer.trainable = False
    """
    base_model.trainable = False        
    model.summary(show_trainable=True)
    
    #x,y=train_ds
    

    metricAccuracy=tf.keras.metrics.CategoricalAccuracy()
    metricLoss=tf.keras.metrics.CategoricalCrossentropy()
    loss=tf.keras.losses.CategoricalCrossentropy()
    monitor_loss="val_categorical_crossentropy"
    monitor_acc="val_categorical_accuracy"
    
    """
    loss=keras.losses.SparseCategoricalCrossentropy()
    metricAccuracy=tf.keras.metrics.CategoricalAccuracy()
    metricLoss=tf.keras.metrics.SparseCategoricalCrossentropy()
    monitor_loss="val_sparse_categorical_crossentropy"
    monitor_acc="val_sparse_categorical_accuracy"
    """
    
    model.compile(optimizer=keras.optimizers.legacy.Adam(),
              loss=loss,
              metrics=[metricAccuracy, metricLoss]
              )
    #model.fit(new_dataset, epochs=20, callbacks=..., validation_data=...)
    
    earlyStopping = keras.callbacks.EarlyStopping(monitor=monitor_loss, patience=10, verbose=1, mode='min')
    mcp_save_loss = keras.callbacks.ModelCheckpoint(path_save_loss, save_best_only=True, save_weights_only=False, monitor=monitor_loss, mode='min')
    #mcp_save_accuracy = keras.callbacks.ModelCheckpoint(path_save_accuracy, save_best_only=True, save_weights_only=False, monitor=monitor_acc, mode='max')
    reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor=monitor_loss, factor=0.1, patience=5, verbose=1, mode='min')
    
    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    model.fit(
            train_ds,
            #x,
            #y,
            batch_size=32,
            epochs=15,
            validation_data=val_ds,
            #validation_split=0.1,
            shuffle=True,
            initial_epoch=0,
            validation_freq=1,
            verbose=2,
            workers=num_workers,
            callbacks=[earlyStopping, reduce_lr_loss, mcp_save_loss]# mcp_save_accuracy, reduce_lr_loss]
            )

    # Unfreeze the base model
    
    """
    for layer in base_model.layers:
        layer.trainable = True
    """
    base_model.trainable = True
    
    model=tf.keras.models.load_model(path_save_loss)
    model.trainable= True
    model.summary(show_trainable=True)
    
    #base_model.trainable = True
    #model.summary(show_trainable=True)
    
    loss=tf.keras.losses.CategoricalCrossentropy()
    metricAccuracy=tf.keras.metrics.CategoricalAccuracy()
    metricLoss=tf.keras.metrics.CategoricalCrossentropy()
    monitor_loss="val_categorical_crossentropy"
    monitor_acc="val_categorical_accuracy"
    
    """
    loss=keras.losses.SparseCategoricalCrossentropy()
    metricAccuracy=tf.keras.metrics.SparseCategoricalAccuracy()
    metricLoss=tf.keras.metrics.SparseCategoricalCrossentropy()
    monitor_loss="val_sparse_categorical_crossentropy"
    monitor_acc="val_sparse_categorical_accuracy"
    """
    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are take into account
    model.compile(optimizer=keras.optimizers.legacy.Adam(1e-4),  # Very low learning rate
                loss=loss,
                metrics=[metricAccuracy, metricLoss]
                )
    
    #earlyStopping = keras.callbacks.EarlyStopping(monitor=monitor_loss, patience=10, verbose=1, mode='min')
    mcp_save_loss = keras.callbacks.ModelCheckpoint(path_save_loss, save_best_only=True, save_weights_only=False, monitor=monitor_loss, mode='min')
    mcp_save_accuracy = keras.callbacks.ModelCheckpoint(path_save_accuracy, save_best_only=True, save_weights_only=False, monitor=monitor_acc, mode='max')
    reduce_lr_loss = keras.callbacks.ReduceLROnPlateau(monitor=monitor_loss, factor=0.1, patience=5, verbose=1, mode='min')
    
    model.fit(
            train_ds,
            #x,
            #y,
            batch_size=32,
            epochs=40,
            validation_data=val_ds,
            #validation_split=0.1,
            shuffle=True,
            initial_epoch=0,
            validation_freq=1,
            verbose=2,
            workers=num_workers,
            callbacks=[earlyStopping, reduce_lr_loss, mcp_save_loss, mcp_save_accuracy]
            )

def load_data(image_size,batch_size):
    """
    mean = np.array([123.68, 116.779, 103.939], dtype="float32")
    #ResNet18, scale_layer = Classifiers.get('resnet18')
    # initialize the training data augmentation object
    trainAug = ImageDataGenerator(
        #featurewise_center=True, 
        #samplewise_center=True,
        #rescale=2./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.3, 1.0),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        #preprocessing_function=scale_layer,
        )
    # initialize the validation/testing data augmentation object (which
    # we'll be adding mean subtraction to)
    valAug = ImageDataGenerator()
    testAug = ImageDataGenerator()
    # define the ImageNet mean subtraction (in RGB order) and set the
    # the mean subtraction value for each of the data augmentation
    # objects
    
    trainAug.mean = mean
    valAug.mean = mean
    #trainAug.std = 64.
    
    train_ds = trainAug.flow_from_directory(
        directory=PATH_TRAIN_MODELS_FOLDER,
        #classes='inferred',
        class_mode='categorical',
        batch_size=batch_size,
        target_size=(image_size[0],image_size[1])
    )
    val_ds = valAug.flow_from_directory(
        directory=PATH_VAL_MODELS_FOLDER,
        #classes='inferred',
        class_mode='categorical',
        batch_size=batch_size,
        target_size=(image_size[0],image_size[1])
    )
    test_ds = testAug.flow_from_directory(
        directory=PATH_TEST_FOLDER,
        #labels='inferred',
        class_mode='categorical',
        batch_size=batch_size,
        target_size=(image_size[0],image_size[1])
    )
    
    """
    """
    train_ds=tf.keras.preprocessing.image_dataset_from_directory(
        PATH_TRAIN_MODELS_FOLDER,
        labels='inferred',
        label_mode='categorical',
        #class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(image_size[0],image_size[1]),
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
    
    val_ds=tf.keras.preprocessing.image_dataset_from_directory(
        PATH_VAL_MODELS_FOLDER,
        labels='inferred',
        label_mode='categorical',
        #class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(image_size[0],image_size[1]),
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
    
    test_ds=tf.keras.preprocessing.image_dataset_from_directory(
        PATH_TEST_FOLDER,
        labels='inferred',
        label_mode='categorical',
        #class_names=None,
        color_mode='rgb',
        batch_size=batch_size,
        image_size=(image_size[0],image_size[1]),
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
    """
    
    
    train_ds = keras.utils.image_dataset_from_directory(
        directory=PATH_TRAIN_MODELS_FOLDER,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(image_size[0],image_size[1]))
    val_ds = keras.utils.image_dataset_from_directory(
        directory=PATH_VAL_MODELS_FOLDER,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(image_size[0],image_size[1]))
    test_ds = keras.utils.image_dataset_from_directory(
        directory=PATH_TEST_FOLDER,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=(image_size[0],image_size[1]))
    
    return train_ds, val_ds, test_ds
    
if __name__ == "__main__":
    """
    def fix_gpu():
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)


    fix_gpu()
    """
    #FLAGS
    train=True
    test=False
    #input_shape=(32,32,3)
    input_shape=(362, 512, 3)
    #input_shape=(224, 224, 3)
    batch_size=32
    num_workers=6
    #MODEL_NAME="MobileNetV3S"
    #MODEL_NAME="VGG16"
    #MODEL_NAME="ResNet18"
    #MODEL_NAME="EfficientNetB7"
    #MODEL_NAME="InceptionResNetV2"
    
    
    #PATHS to save models
    #path_model_loss=PATH_TO_SAVE_MODELS+"/"+MODEL_NAME+"/cfar100best_loss.h5"
    #path_model_accuracy=PATH_TO_SAVE_MODELS+"/"+MODEL_NAME+"/cfar100best_accuracy.h5"
    
    """
    train_ds, val_ds, test_ds=load_data(input_shape, batch_size)
    for images, labels in next(zip(train_ds)):
        img=images
        break
    print(img.shape)
    """
    
    transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    data_augmentation = tf.keras.Sequential([
        #keras_cv.layers.AugMix((0,255),severity=0.3)
        #tf.keras.layers.RandomFlip("horizontal"),
        #tf.keras.layers.RandomRotation(0.2),
        #tf.keras.layers.RandomZoom(0.2),
        #tf.keras.layers.RandomHeight(0.2),
        #tf.keras.layers.RandomWidth(0.2)
   # preprocessing.Rescale()
    ], name="data_augmentation")
    
    # create the base pre-trained model
    #base_model, scale_layer = get_base_model(MODEL_NAME, input_shape)
    
    train_ds, val_ds, test_ds = load_data(input_shape, batch_size)
    #(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    #x_train_model, x_train_selector, y_train_model, y_train_selector = train_test_split(x_train, y_train, test_size=0.5, random_state=11)
    
    #x_test_model, x_test_selector, y_test_model, y_test_selector = train_test_split(x_test, y_test, test_size=0.5, random_state=11)
    #train_ds=(x_train_model, y_train_model)
    
    TO_TRAIN=["VGG16","NASNetMobile"]
    
    
    for MODEL_NAME in TO_TRAIN:
        
        path_model_loss=PATH_TO_SAVE_MODELS+"/"+MODEL_NAME+"/best_loss.h5"
        path_model_accuracy=PATH_TO_SAVE_MODELS+"/"+MODEL_NAME+"/best_accuracy.h5"
        
        model, base_model = get_wrap_model(MODEL_NAME, input_shape, 271)
        #model, base_model = get_wrap_model(MODEL_NAME, input_shape, 10)
        
        match MODEL_NAME:
            case "MobileNetV3S":
                """
                x_train_model=tf.keras.applications.mobilenet_v3.preprocess_input(x_train_model)
                train_ds=(x_train_model,y_train_model)
                val_ds=None
                """
                
                train_ds=train_ds.map(lambda x, y: (
                    tf.keras.applications.mobilenet_v3.preprocess_input(data_augmentation(x)), 
                    y))
                
                #train_ds=train_ds.map(lambda x, y: (
                #    tf.keras.applications.mobilenet_v3.preprocess_input(x), y))
                val_ds= val_ds.map(lambda x, y: (tf.keras.applications.mobilenet_v3.preprocess_input(x), y))
                
            case "NASNetMobile":
                """
                x_train_model=tf.keras.applications.nasnet.preprocess_input(x_train_model)
                train_ds=(x_train_model,y_train_model)
                val_ds=None
                """
                train_ds=train_ds.map(lambda x, y: (
                    keras.applications.nasnet.preprocess_input(data_augmentation(x)), 
                    y))
                #train_ds=train_ds.map(lambda x, y: (
                #    keras.applications.nasnet.preprocess_input(x), y))
                val_ds= val_ds.map(lambda x, y: (keras.applications.nasnet.preprocess_input(x), y))
                
                
            case"ResNet18":
                ResNet18, scale_layer = Classifiers.get('resnet18')
                train_ds=train_ds.map(lambda x, y: (scale_layer(data_augmentation(x)), y))
                #train_ds=train_ds.map(lambda x, y: (scale_layer(x), y))
                val_ds= val_ds.map(lambda x, y: (scale_layer(x), y))
                
            case "VGG16":
                """
                x_train_model=tf.keras.applications.vgg16.preprocess_input(x_train_model)
                train_ds=(x_train_model,y_train_model)
                val_ds=None
                """
                train_ds=train_ds.map(lambda x, y: (
                    keras.applications.vgg16.preprocess_input(data_augmentation(x)),
                    y))
                #train_ds=train_ds.map(lambda x, y: (
                #    keras.applications.vgg16.preprocess_input(x),y))
                val_ds= val_ds.map(lambda x, y: (keras.applications.vgg16.preprocess_input(x), y))
                
                
            case  "EfficientNetB7":
                train_ds=train_ds.map(lambda x, y: (
                    keras.applications.efficientnet.preprocess_input(data_augmentation(x)),
                    y))
                #train_ds=train_ds.map(lambda x, y: (
                #    keras.applications.vgg16.preprocess_input(x),y))
                val_ds= val_ds.map(lambda x, y: (keras.applications.efficientnet.preprocess_input(x), y))
                
            case "InceptionResNetV2":
                train_ds=train_ds.map(lambda x, y: (
                    keras.applications.inception_resnet_v2.preprocess_input(data_augmentation(x)),
                    y))
                #train_ds=train_ds.map(lambda x, y: (
                #    keras.applications.vgg16.preprocess_input(x),y))
                val_ds= val_ds.map(lambda x, y: (keras.applications.inception_resnet_v2.preprocess_input(x), y))
        
        #brightness_layer=tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0, 255))
        #flip_layer=tf.keras.layers.RandomFlip(mode=HORIZONTAL_AND_VERTICAL)
        #zoom_layer=tf.keras.layers.RandomZoom( height_factor=0.2, width_factor=None, fill_mode='nearest', interpolation='bilinear', seed=None ,fill_value=0.0) #data_format=None,)
        #roatation_layer=tf.keras.layers.RandomRotation(factor, fill_mode='reflect', interpolation='bilinear', seed=None, fill_value=0.0, value_range=(0, 255))#data_format=None,)

        #normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)
        #normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1./255)
        #normalized_training_set = train_ds.map(lambda x, y: (normalization_layer(x), y))

        
        if train:
            FitAndFinetuning(base_model, model, train_ds, val_ds, path_model_loss, path_model_accuracy, num_workers)
            #base_model = InceptionV3(weights='imagenet', include_top=False)
        
        else:
            #model.compile()
            model=load_model(path_model_loss)
            score=model.evaluate(test_ds,
                        batch_size=None,
                        verbose=2,
                        )
            
            print("%.2f" %(score*100))