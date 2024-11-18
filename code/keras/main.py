import keras
import os

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
# initialize the validation/testing data augmentation object (which
# we'll be adding mean subtraction to)
valAug = ImageDataGenerator()
# define the ImageNet mean subtraction (in RGB order) and set the
# the mean subtraction value for each of the data augmentation
# objects
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean


""" DATASET FOLDER """
PATH_DATASET_FOLDER = os.getcwd() + "/../dataset/SUN397/"

""" SELECTOR SUB FOLDERS """
PATH_TRAIN_FOLDER=PATH_DATASET_FOLDER+  "train_selector"
PATH_VAL_FOLDER=PATH_DATASET_FOLDER+"/val_selector"

""" MODELS SUB FOLDERS """
PATH_TRAIN_MODELS_FOLDER=PATH_DATASET_FOLDER+"/val_models"
PATH_VAL_MODELS_FOLDER=PATH_DATASET_FOLDER+"/val_models"

""" TEST PATH FOLDERS """
PATH_TEST_FOLDER=PATH_DATASET_FOLDER+"/test"

train_ds = keras.utils.image_dataset_from_directory(
    directory='training_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))
validation_ds = keras.utils.image_dataset_from_directory(
    directory='validation_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))
test_ds = keras.utils.image_dataset_from_directory(
    directory='validation_data/',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(256, 256))
model = keras.applications.Xception(
    weights=None, input_shape=(256, 256, 3), classes=10)

model.compile(
    optimizer="rmsprop",
    loss='categorical_crossentropy',
    loss_weights=None,
    metrics=None,
    weighted_metrics=None,
    run_eagerly=False,
    steps_per_execution=1,
    jit_compile="auto",
    auto_scale_loss=True,
)
"""
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.FalseNegatives(),
    ],
)
"""

model.fit(
    train_ds,
    batch_size=None,
    epochs=10,
    verbose=2,
    #callbacks=None,
    #validation_split=0.0,
    validation_data=validation_ds,
    shuffle=True,
    #class_weight=None,
    #sample_weight=None,
    initial_epoch=0,
    #steps_per_epoch=None,
    #validation_steps=None,
    #validation_batch_size=None,
    validation_freq=1,
)

model.evaluate(
    test_ds,
    batch_size=None,
    verbose="auto",
    sample_weight=None,
    steps=None,
    callbacks=None,
    return_dict=False,
    #**kwargs
)
