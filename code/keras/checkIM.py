import glob
import os
import tensorflow as tf

""" DATASET FOLDER """
PATH_DATASET_FOLDER = os.getcwd() + "/../../dataset/SUN397"

""" MODELS SUB FOLDERS """
PATH_TRAIN_MODELS_FOLDER=PATH_DATASET_FOLDER+"/train_models"
PATH_VAL_MODELS_FOLDER=PATH_DATASET_FOLDER+"/val_models"

""" TEST PATH FOLDERS """
PATH_TEST_FOLDER=PATH_DATASET_FOLDER+"/test"

""" PATH TO SAVE MODEL """
PATH_TO_SAVE_MODELS="./models"

l=[PATH_TEST_FOLDER]

for path in l:
  img_paths = glob.glob(os.path.join(path,'*/*.*')) # assuming you point to the directory containing the label folders.

  bad_paths = []

  for image_path in img_paths:
      try:
        img_bytes = tf.io.read_file(image_path)
        decoded_img = tf.io.decode_image(img_bytes)
      except tf.errors.InvalidArgumentError as e:
        print(f"Found bad path {image_path}...{e}")
        bad_paths.append(image_path)

      #print(f"{image_path}: OK")

  print("BAD PATHS:")
  for bad_path in bad_paths:
      print(f"{bad_path}")
