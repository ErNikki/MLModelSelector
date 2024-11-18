import torch
from torch import nn
import torchvision
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
import operator

"""
COME FUNZIONA ILOC CON DataFrame DI PANDAS:
iloc permette di accedere alle righe del dataframe quindi iloc[0] restituisce la prima riga e iloc[0:3] le prime tre righe (0,1,2) non è compresa 3!!
però permette anche di accedere alle colonne nel formato [start_row:end_row , start_col:end_col] quindi iloc[0,1] accede riga  1 colonna 1

PERCHè USO I PANDA DATAFRAME?
perchè cosi posso sfruttare mode()
"""
#mode = "hard " or "soft"
def predictions_voting_hard_soft(dl_model, test_loader, mode):#, input_dim):
    #pred_hard, pred_soft = [], []
    pred=[]
    true_labels=[]
    #correct_count, all_count = 0,0
    dl_model.eval()
    #im_dim = transform_obj.transforms[1].size[0]
    with torch.no_grad():
        
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            for i in range(len(labels)):
                img = images[i].unsqueeze(0)#.view(1, 3, input_dim[1], input_dim[2])
                # soft voting
                if mode=="soft":
                    output = dl_model(img)
                    sm = nn.Softmax(dim=1)
                    probabilities = sm(output)
                    prob_arr = (probabilities.detach().cpu().numpy())[0]
                    pred.append(prob_arr)
                
                # hard voting
                else:
                    logps = dl_model(img)
                    #serve davvero questo passaggio?
                    ps = torch.exp(logps)
                    probab = list(ps.cpu()[0])
                    pred_label = probab.index(max(probab))
                    pred.append(pred_label)
                
                
                #exporting to dataframe
                true_label = labels.cpu()[i].item()
                true_labels.append(true_label)
            
                
                """
                if(true_label == pred_label):
                    correct_count += 1
                all_count += 1

    print("Number of images Tested=", all_count)
    print("Model Accuracy=",(correct_count/all_count)*100)
    print("\n")"""
    #return pred_hard, pred_soft
    return pred,true_labels


def hard_voting(models,test_loader):
    models_pred=[]
    true_labels_list=[]
    for model in models:
        pred,true_labels=predictions_voting_hard_soft(model, test_loader, "hard")
        models_pred.append(pred)
        #print(np.asarray(pred).shape)
        if not true_labels_list:
            true_labels_list=true_labels
            
    df_hard_voting = pd.DataFrame(models_pred)
    ensemble_hard_predictions = np.asarray(df_hard_voting.mode(axis=0).iloc[0])
    ensemble_hard_score = accuracy_score(np.asarray(true_labels_list), ensemble_hard_predictions)
    print(f"The Accuracy Score of Hard Voting Ensemble is:  {(ensemble_hard_score*100):.4f} %")
    return ensemble_hard_predictions


def soft_voting(models, test_loader,num_classes):
    models_pred=[]
    true_labels_list=[]
    for model in models:
        pred,true_labels=predictions_voting_hard_soft(model, test_loader, "soft")
        models_pred.append(pred)
        if not true_labels_list:
            true_labels_list=true_labels
            
    df_soft_voting = pd.DataFrame(models_pred)
    ensemble_soft_preds = []
    for y in range(len(df_soft_voting.columns)):
        sample = tuple([0.0]*num_classes)
        for x in range(len(df_soft_voting)):
            sample = tuple(map(operator.add, sample, (tuple(df_soft_voting.iloc[x,y]))))
        sample = tuple(ti/len(sample) for ti in sample)
        element = max(sample)
        idx = sample.index(element)
        ensemble_soft_preds.append(idx)
    
    ensemble_soft_score = accuracy_score(np.asarray(true_labels), np.asarray(ensemble_soft_preds))
    print(f"The Accuracy Score of Soft Voting Ensemble is:  {(ensemble_soft_score*100):.4f} %")
    return ensemble_soft_preds

        
    
    