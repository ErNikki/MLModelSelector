import torch
from torch import nn
import torchvision
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
import operator
from collections import Counter
import time

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
    total_inference_time=0
    n_samples=0
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        for i in range(len(labels)):
            image=images[i]
            label=labels.cpu()[i].item()
            
            torch.cuda.synchronize()
            start_time = time.time()  # Inizio del conteggio
            pred=hard_voting(models, image)
            torch.cuda.synchronize()
            inference_time = time.time() - start_time  # Fine del conteggio
            total_inference_time+=inference_time
            
            hard_pred.append(pred)
            true_labels.append(label)
            n_samples+=1

    ensemble_hard_score = accuracy_score(np.asarray(true_labels), np.asarray(hard_pred))
    print(f"numeber of samples: {(n_samples)}")
    print(f"The Accuracy Score of Hard Voting Ensemble is:  {(ensemble_hard_score*100):.4f} %")
    print(f"Inference time total: {total_inference_time*1000}, mean: {(total_inference_time/n_samples)*1000}")
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
    total_inference_time=0
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)
        for i in range(len(labels)):
            image=images[i]
            label=labels.cpu()[i].item()
            
            torch.cuda.synchronize()
            start_time = time.time()  # Inizio del conteggio
            pred=soft_voting(models, image)
            torch.cuda.synchronize()
            inference_time = time.time() - start_time  # Fine del conteggio
            total_inference_time+=inference_time
            
            hard_pred.append(pred)
            true_labels.append(label)
            n_samples+=1

    ensemble_soft_score = accuracy_score(np.asarray(true_labels), np.asarray(hard_pred))
    print(f"numeber of samples: {(n_samples)}")
    print(f"The Accuracy Score of Soft Voting Ensemble is:  {(ensemble_soft_score*100):.4f} %")
    print(f"Inference time total: {total_inference_time*1000}, mean: {(total_inference_time*1000/n_samples)}")


