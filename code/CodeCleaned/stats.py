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

import matplotlib.pyplot as plt

"""
STATS
"""
FOLDER_STATS_PATH="./stats/"
TEST_FOLDER_STATS="Test_set/"
VAL_FOLDER_STATS="Validation_set/"
TRAIN_SELECTOR_FOLDER_STATS="Train_selector_set/"
#USED IN CODE
CHOSEN_FOLDER_STATS= FOLDER_STATS_PATH+ TRAIN_SELECTOR_FOLDER_STATS

"""
DATASET
"""
PATH_DATASET_FOLDER = os.getcwd() + "/../../dataset/SUN397/"
PATH_TRAIN_FOLDER=PATH_DATASET_FOLDER+  "train_selector"
PATH_TEST_FOLDER=PATH_DATASET_FOLDER+"/test"
PATH_VAL_FOLDER=PATH_DATASET_FOLDER+"/val_selector"
#USED IN CODE
CHOSEN_PATH= PATH_VAL_FOLDER

def make_charts(models_names, total_number_images, accuracies, unique_predictions, predictions_counts):
    
    def create_folder_if_not_exists(folder_path):
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
    create_folder_if_not_exists(CHOSEN_FOLDER_STATS)
    
    # Data for image predictions
    prediction_labels = ['0 times', '1 time', '2 times', '3 times']
    # Calcola il numero totale di immagini uniche
    total_unique_images = sum(unique_predictions)

    # Bar chart for accuracies
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models_names, accuracies, color=['blue', 'green', 'red'])
    plt.title('Model Accuracies')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Models')
    plt.ylim(0, 100)

    # Add labels above the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}%', ha='center', va='bottom')

    plt.savefig(os.path.join(CHOSEN_FOLDER_STATS, 'model_accuracies.png'))
    plt.show()

    # Histogram of the number of predicted images
    plt.figure(figsize=(10, 6))
    bars = plt.bar(prediction_labels, predictions_counts, color='purple')
    plt.title('Number of times a single image as been predicted')
    plt.ylabel('Number of Images')
    plt.xlabel('Times single Images Were Predicted')

    # Add labels above the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')

    plt.savefig(os.path.join(CHOSEN_FOLDER_STATS, 'number_of_images_predicted.png'))
    plt.show()

    # Pie chart for unique predicted images
    plt.figure(figsize=(8, 8))
    plt.pie(unique_predictions, labels=models_names, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen', 'salmon'])
    plt.title(f'Distribution of Unique Images Predicted (Total: {total_unique_images} over :{total_number_images})')
    plt.savefig(os.path.join(CHOSEN_FOLDER_STATS ,'unique_images_predicted.png'))
    plt.show()

    
def predict(model, image):
    """
    #versione probailistica
    
    img=image.unsqueeze(0)
    logps = model(img)
    print(logps)
    #serve davvero questo passaggio?
    ps = torch.exp(logps)
    probab = list(ps.cpu()[0])
    pred_label = probab.index(max(probab))
    #pred.append(pred_label)
    return pred_label
    """
    
    img = image.unsqueeze(0)  # Aggiungi una dimensione batch
    logps = model(img)        # Ottieni i logits dal modello
    #logits = logps.cpu()[0]   # Porta i logits sulla CPU e prendi il primo batch
    #pred_label = logits.argmax().item()  # Trova l'indice del logit più alto (classe predetta) 
    _, preds = torch.max(logps, dim=1)
    return preds.item()
    


def get_predictions_list(models, test_loader, device):
    true_labels=[]
    models_pred={}
    correct_predictions = {}
    
    # Inizializza il contatore per le predizioni corrette per ogni modello
    for index_model in range(len(models)):
        correct_predictions[index_model] = 0
        
    total_images = 0  # Contatore per il numero totale di immagini
    
    for images, labels in test_loader:
        images=images.to(device)
        labels=labels.to(device)

        for i in range(len(labels)):
            image=images[i]
            true_label=labels.cpu()[i].item()
            index_model=0
            with torch.no_grad():
                for model in models:
                    #model=models[model_key]
                    model.eval()
                    pred=predict(model,image)
                    
                     # Verifica se la predizione è corretta
                    if pred == true_label:
                        correct_predictions[index_model] += 1
                        
                    if index_model not in models_pred:
                        models_pred[index_model]=[]
                    models_pred[index_model].append(pred)
                    index_model+=1

            true_labels.append(true_label)
            total_images += 1  # Incrementa il conteggio delle immagini

    return total_images, correct_predictions, true_labels, models_pred

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

    test_data=Sun397(CHOSEN_PATH, TRANSFORM_IMG)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,  num_workers =4)

    #MODELS_NAME=["MobNetV3S","ResNet18","ShufflNetV2_x05"]
    MODELS_NAME=["GoogLeNet","ResNet18","MobNetV3S"]
    #MODELS_NAME=["Efficientnet_b3"]
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

    total_images, correct_predictions, true_labels, models_pred = get_predictions_list(models_loaded, test_data_loader, device)

    exactly_predictions=[]
    for i in range(len(models_pred)+1):
        #exactly_predictions[i] = 0
        exactly_predictions.append(0)

    unique_predictions=[]
    for i in range(len(models_pred)):
        #unique_predictions[i] = 0
        unique_predictions.append(0)
        
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
    
     
    print("")
    print("total number of images : " + str(total_images))
    print("")
    accuracies=[]
    for index_model in correct_predictions:
        print(models_name_to_dict[index_model] + f": {correct_predictions[index_model]} correct predictions out of {total_images} with percentage {correct_predictions[index_model]/total_images*100}")
        accuracies.append(correct_predictions[index_model]/total_images*100)
    
    print("")
    #predictions_counts=[]
    for index in range(len(MODELS_NAME)+1):
        print(str(exactly_predictions[index]) +" images has been predicted " + str(index) +" times")
        #predictions_counts.append(exactly_predictions[index])
        
    print("")
    #unique_predictions=[]
    for index in range(len(MODELS_NAME)):
        print(models_name_to_dict[index] + " : has predicted " + str(unique_predictions[index]) + " unique images")
        #unique_predictions.append(unique_predictions[index])
    print("")
    
    make_charts(MODELS_NAME, total_images ,accuracies, unique_predictions, exactly_predictions)  
    