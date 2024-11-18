import random
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision 
import os

class Sun397(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        self.data = [img for img in Path(img_dir).glob("*/*.jpg")]
        self.labels=[]

        #classi le cui immagini non vengono lette
        
        self.labels_to_remove=[]
        """
        for img in Path(img_dir).glob("*/*/*/*.jpg"):
            if not  img.parent.parent.name in self.labels_to_remove:
                self.labels_to_remove.append(img.parent.parent.name)
        """
        #labels che non tengono conto delle classi con pattern */*/*
        for img in Path(img_dir).glob("*"):
            if not img.name in self.labels_to_remove:
                self.labels.append(img.name)
        
                

        self.labels_dict = self.labels_to_dict(self.labels)
        self.num_classes = len(self.labels_dict)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(str(img_path)).convert("RGB")
        label = self.labels_dict[img_path.parent.name]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
        
    def labels_to_dict(self, list_of_labels):
        labels = sorted(list(set(list_of_labels)))

        labels_dict = {}
        for idx in range(len(labels)):
            labels_dict[labels[idx]] = idx

        return labels_dict

"""
PATH_DATASET_FOLDER = os.getcwd() + "/../dataset/SUN397/"
PATH_TRAIN_FOLDER=PATH_DATASET_FOLDER+  "train_selector"
PATH_TEST_FOLDER=PATH_DATASET_FOLDER+"/test"
PATH_VAL_FOLDER=PATH_DATASET_FOLDER+"/val_selector"
PATH_VAL_MODELS_FOLDER=PATH_DATASET_FOLDER+"/val_models"

TRAIN_DATA = PATH_DATASET_FOLDER+"train_models"
VAL_DATA = PATH_DATASET_FOLDER+"val_models"
#TEST_DATA = "/test"

TRANSFORM_IMG = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(256),
        #torchvision.transforms.CenterCrop(256),
        torchvision.transforms.Resize((362, 512), antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )
        ])


test_data=Sun397(PATH_TEST_FOLDER,TRANSFORM_IMG)
train_data=Sun397(PATH_TRAIN_FOLDER,TRANSFORM_IMG)
val_data=Sun397(PATH_VAL_FOLDER,TRANSFORM_IMG)

model_data=Sun397(TRAIN_DATA,TRANSFORM_IMG)
model_val_data=Sun397(VAL_DATA,TRANSFORM_IMG)

print("#########################")
print(test_data.labels_to_remove)
print(test_data.num_classes)


print("#########################")
print(train_data.labels_to_remove)
print(train_data.num_classes)

print("#########################")
print(val_data.labels_to_remove)
print(val_data.num_classes)

print("#########################")
print(model_val_data.labels_to_remove)
print(model_val_data.num_classes)

print("#########################")
print(model_val_data.labels_to_remove)
print(model_val_data.num_classes)
"""