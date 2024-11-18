import torch
from torch import nn
import torchvision
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
import operator
from collections import Counter
from utils import hardware_check, load_model
import os
from SUN397 import Sun397
from pathlib import Path
from collections import Counter
from Param import *
def predict(model, image):
    img=image.unsqueeze(0)
    logps = model(img)
    #serve davvero questo passaggio?
    ps = torch.exp(logps)
    probab = list(ps.cpu()[0])
    pred_label = probab.index(max(probab))
    #pred.append(pred_label)
    return pred_label

def get_predictions_list(models, test_loader, device):
    true_labels=[]
    models_pred={}
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)

        for i in range(len(labels)):
            image=images[i]
            label=labels.cpu()[i].item()
            index_model=0
            with torch.no_grad():
                for model in models:
                    #model=models[model_key]
                    model.eval()
                    pred=predict(model,image)
                    if index_model not in models_pred:
                        models_pred[index_model]=[]
                    models_pred[index_model].append(pred)
                    index_model+=1

            true_labels.append(label)

    return true_labels, models_pred

if __name__ == "__main__":

    device = hardware_check()
    
    TRANSFORM_IMG = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(256),
        #torchvision.transforms.CenterCrop(256),
        torchvision.transforms.Resize((362, 512), antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )
        ])

    train_data=Sun397(PATH_TRAIN_FOLDER,TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True,  num_workers =1, prefetch_factor=64)
        
    test_data=Sun397(PATH_TEST_FOLDER,TRANSFORM_IMG)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True,  num_workers =4)
    
    val_data=Sun397(PATH_VAL_FOLDER,TRANSFORM_IMG)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True,  num_workers =4, prefetch_factor=16)

    MODELS_NAME=[ "GoogLeNet","ResNet18","MobNetV3S"] 
    env_path =Path("../models/")

    models_loaded=[]
    models_name_to_dict=[]
    index=0
    for model_name in MODELS_NAME:
        pth_path_model = Path(env_path, model_name)
        pth_path_model = os.path.join(pth_path_model, "best_valAcc_model.pth")
        model= load_model(model_name, pth_path_model, device)
        models_loaded.append(model)
        models_name_to_dict.append(model_name)

    true_labels, models_pred=get_predictions_list(models_loaded, val_data_loader, device)

    exactly_predictions={}
    for i in range(len(models_pred)+1):
        exactly_predictions[i] = 0

    unique_predictions={}
    for i in range(len(models_pred)):
        unique_predictions[i] = 0
    
    for index_label in range(len(true_labels)):

        label = true_labels[index_label]
        list_predicted_label = []
        for index_model in range(len(models_pred)):
            predicted_label=models_pred[index_model][index_label]
            list_predicted_label.append(predicted_label)

        count=Counter(list_predicted_label)[label]

        exactly_predictions[count]+=1
        if count==1:
            #print(list_predicted_label.index(label))
            unique_predictions[list_predicted_label.index(label)]+=1

    
    print("exactly : " + str(exactly_predictions))
    print("unique_predictions : " + str(unique_predictions))