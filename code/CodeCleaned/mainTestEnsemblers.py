import torch
from torch import nn
import torchvision
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
import operator
from collections import Counter
from Param import *
from pathlib import Path
import os
from utils import hardware_check,load_model
from SUN397 import Sun397

def pred_voting_hard(model, image):
    img=image.unsqueeze(0)
    logps = model(img)
    #serve davvero questo passaggio?
    ps = torch.exp(logps)
    probab = list(ps.cpu()[0])
    pred_label = probab.index(max(probab))
    #pred.append(pred_label)
    return pred_label

def hard_voting(models, image):
    models_pred=[]
    with torch.no_grad():
        for model in models:
            model.eval()
            pred=pred_voting_hard(model,image)
            models_pred.append(pred)

    return Counter(models_pred).most_common(1)[0][0]

def hard_voting_test_loader(models, test_loader, device):
    true_labels=[]
    hard_pred=[]
    n_samples=0
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        for i in range(len(labels)):
            image=images[i]
            label=labels.cpu()[i].item()
            pred=hard_voting(models, image)
            hard_pred.append(pred)
            true_labels.append(label)
            n_samples+=1

    ensemble_hard_score = accuracy_score(np.asarray(true_labels), np.asarray(hard_pred))
    print(f"numeber of samples: {(n_samples)}")
    print(f"The Accuracy Score of Hard Voting Ensemble is:  {(ensemble_hard_score*100):.4f} %")
    return hard_pred

def pred_soft_voting(model,image):
    img=image.unsqueeze(0)
    output = model(img)
    sm = nn.Softmax(dim=1)
    probabilities = sm(output)
    prob_arr = (probabilities.detach().cpu().numpy())[0]
    return prob_arr

def soft_voting(models,image):
    models_pred=[]
    with torch.no_grad():
        for model in models:
            model.eval()
            pred=pred_soft_voting(model,image)
            models_pred.append(pred)
    
    num_classes=len(models_pred[0])
    sample = tuple([0.0]*num_classes)
    for pred in models_pred:
        sample = tuple(map(operator.add, sample, pred ))
    sample = tuple(ti/len(sample) for ti in sample)
    element = max(sample)
    pred = sample.index(element)

    return pred

def soft_voting_test_loader(models, test_loader, device):
    true_labels=[]
    hard_pred=[]
    n_samples=0
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        for i in range(len(labels)):
            image=images[i]
            label=labels.cpu()[i].item()
            pred=soft_voting(models, image)
            hard_pred.append(pred)
            true_labels.append(label)
            n_samples+=1

    ensemble_soft_score = accuracy_score(np.asarray(true_labels), np.asarray(hard_pred))
    print(f"The Accuracy Score of Soft Voting Ensemble is:  {(ensemble_soft_score*100):.4f} %")


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

    test_data=Sun397(PATH_TEST_FOLDER, TRANSFORM_IMG)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,  num_workers =4)
    
    val_data=Sun397(PATH_VAL_FOLDER, TRANSFORM_IMG)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,  num_workers =4)

    #anche questa può essere una variabile in init
    env_path =Path("../models/")

    #MODELS_NAME=["ShufflNetV2_x05","ResNet18","MobNetV3S"]
    MODELS_NAME=["ResNet18","MobNetV3S","GoogLeNet"]
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

    
    hard_voting_test_loader(models_loaded, val_data_loader, device)
    soft_voting_test_loader(models_loaded, val_data_loader, device)