import torch
import torchvision
from torchvision.transforms import v2

from utils import hardware_check, to_device, build_model, load_model
#from models import ModelLoader
#from datasets import Sun397
#import matplotlib.pyplot as plt
import numpy as np

from gym import Env
from gym.spaces import Discrete, Box

import os

from pathlib import Path

from itertools import cycle

import matplotlib.pyplot as plt
#from PIL import Image
#from collections import Counter

#num of frame per episode forse dopo sarebbe meglio farne una randomica con tot episodi
#FRAME_PER_EPISODE=512

#APPLICARE unsqueeze(1) IN RETURN COSI DA AVERE 1X84X84 RICORDA CHE INPUT DI CONV2D è IL NUMERO DEI CANALI IN INPUT!! VA LEVATO SE POI SI DECIDE DI NON USARE 84X84

class newTestEnvSimplified(Env):
    #def __init__(self,model1,model2,model3):
    def __init__(self, models, test_data_loader, device, frame_per_episode, models_path=Path("../models/")):
        
        #Used inside train to specify name of file
        self.spec_id="netSelector"
        
        #device used to process images
        self.device = device
        
        """
        STATS VARIABLES
        """
        #track actions
        self.track_actions=[]
        #track right guesses of models 
        self.track_right_actions=[]
        #track models name associated with index to better display
        self.models_name_to_dict=[]
        
        
        #save dataloader in order to re-shuffle it at the end of all batches
        #self.train_data_loader=train_data_loader    
        #build iterable to iterate on dataloader
        self.actual_iterable=cycle(iter(test_data_loader))
        
        
        """
        TAKE CARE OF FRAMES IN THE EPISODE
        """
        #used to stop episode after tot framed processed
        self.frame_per_episode=frame_per_episode
        #count of many frames are processed during episode in order to end it if overcome frame_per_episode 
        self.actual_frame=0
        
        """
        TAKE CARE OF ACTUAL IMAGE PROCESSED
        """
        #Used to store current bash of images beloning to the actual episode
        self.images=None
        self.labels=None 
        #current image over wich perfor actions
        self.actual_image=None
        self.actual_label=None
        
        #take couunt of actual episode reward
        self.episode_reward=0
        
        #used to give about the number of actions of the environment
        self.action_space = len(models)
        
        #used to give about the dimension of the images processed by env
        #self.observation_space = Box(0, 255, (1, 84, 84), np.uint8)
        self.observation_space = Box(0,255,(3,362,512),np.uint8)

        """
        LOADING MODELS
        """
        
        self.models_loaded=[]
        
        #DA USARE COME VARIABILE INIT
        
        self.accuracy=0
        self.models_name=models
        #index=0
        for model_name in models:
            pth_path_model = Path(models_path, model_name)
            pth_path_model = os.path.join(pth_path_model, "best_valAcc_model.pth")
            print(pth_path_model)
            model= load_model(model_name, pth_path_model, self.device)
            self.models_loaded.append(model)
            self.models_name_to_dict.append(model_name)
        
        for i in range(0,self.action_space):
            self.track_actions.append(0)
            self.track_right_actions.append(0)
            #index+=1
        #print(self.models_loaded)
        
    
    def step(self, action):     
        
        """
        unsqueeze
        """
        #FROM [3,362,512] TO [1,3,362,512] in order to process the image using agent model
        img = self.actual_image.unsqueeze(0).to(self.device)
        label = self.actual_label
        
    
        """
        image processing
        """
        #Usually the action choosen random is an int, the action choosen by the model is a tensor
        if not (isinstance(action, int)):
            action=action.detach().cpu().item()
        
        
        predictions=[]        
        for i in range(0, len(self.models_name)):
            
            #based on the number of the action choose a model
            model=self.models_loaded[i]
            #process image in order to get predicted label by the chosen model
            with torch.no_grad():
                pred = model(img)
                pred = pred.detach().cpu().numpy()
                predicted_label=np.argmax(pred)
                
            predictions.append(predicted_label)
            
        #stats to track action taken in a single episode by every model
        #self.track_actions[action]+=1       
        
        """
        COMPUTING REWARD
        """
        self.track_actions[action]+=1
        match action:
            
            #primo modello unico
            case 0:
                
                if predictions[0]==label:
                    reward=0.3
                    self.accuracy+=1
                    self.track_right_actions[0]+=1
                else:
                    reward=-1
                
            case 1:
                
                if label==predictions[1]:
                    if predictions.count(label)==1:
                        reward=1
                        self.accuracy+=1
                        self.track_right_actions[1]+=1
                    else:
                        reward=0.2
                        self.accuracy+=1
                        self.track_right_actions[1]+=1
                else:
                    reward=-1
                    """
                    done=True
                    self.actual_frame+=1
                    self.actual_image = self.images[self.actual_frame]
                    self.actual_label = self.labels[self.actual_frame]
                    return self.actual_image, reward, done, 0 
                    """  
                    
            case 2:
               
                if label==predictions[2]:
                    if predictions.count(label)==1:
                        reward=1
                        self.accuracy+=1
                        self.track_right_actions[2]+=1
                    else:
                        reward=0.2
                        self.accuracy+=1
                        self.track_right_actions[2]+=1
                    
                else:
                    reward=-1
                    """
                    done=True
                    self.actual_frame+=1
                    self.actual_image = self.images[self.actual_frame]
                    self.actual_label = self.labels[self.actual_frame]
                    return self.actual_image, reward, done, 0 
                    """
        """
        match action:

            #primo modello unico
            case 0:
                
                if label==predictions[0]:
                    self.accuracy+=1
                    self.track_right_actions[0]+=1
                    if predictions.count(label)==1:
                        reward=1
                    else:
                        reward=-1
                    
                else:
                    reward=-1
                

            case 1:
                
                if label==predictions[1]:
                    self.accuracy+=1
                    self.track_right_actions[1]+=1
                    if predictions.count(label)==1:
                        reward=1
                    else:
                        reward=-1
                else:
                    reward=-1
                
            case 2:
                
                if predictions[2]==label:
                    self.accuracy+=1
                    self.track_right_actions[2]+=1
                    
                reward=-0.3
                
        """
        self.episode_reward+=reward
       
            
            
        #stats to track RIGHT action taken in a single episode by every model
        #self.track_right_actions_per_model[action]+=1
        
        """
        checking if all frames has been processed
        """
        done=False
        if(self.actual_frame == self.frame_per_episode-2):
            done=True
            #reward=self.episode_reward
        
        """
        try to compute new state (next image)
        """
        #computo il nuovo stato
        #ricorda che self.actual_image ha dimensione (1024,3,362,1512) quindi se prendo il primo elemento ottengo (3,260,160) 
        
        try:
            #if there exists another image inside batch load it
            self.actual_frame+=1
            self.actual_image = self.images[self.actual_frame]
            self.actual_label = self.labels[self.actual_frame]
            return self.actual_image, reward, done, 0
            
        except:
            #else end episode and create new iterable to shuffle dataloader
            done=True
            del self.actual_iterable
            self.actual_iterable=iter(self.train_data_loader)
            return self.actual_image, reward, done, 0                
            

    def render(self):
    # not needed 
        pass
    
    def reset(self):
        
        #setup episode reward counter and actual frame (state) counter of the environment
        self.episode_reward=0
        self.actual_frame=0
        self.accuracy=0
        
        """
        TRACK STATS
        """
        self.track_actions=[]
        self.models_name_to_dict=[]
        self.track_right_actions=[]     
        
        for model_name in self.models_name:
            self.models_name_to_dict.append(model_name)
                      
        for i in range(0,self.action_space):
            self.track_actions.append(0)
            self.track_right_actions.append(0)
        
        """
        READ NEW BATCH OF IMAGES
        """
            
        #read netx bash of images
        self.images, self.labels = next(self.actual_iterable)
        
        #get first image "start state"
        self.actual_image = self.images[self.actual_frame]
        self.actual_label = self.labels[self.actual_frame]

        return self.actual_image
    
    #usato nel train ma qui inutile
    def close(self, print_stats=(False,"")):

        """
        PRINT COLLECTED STATS
        """
        print()
        print("ACTION CHOOSEN:")
        print(self.track_actions)
        
        print()
        print("RIGHT CHOOSEN:")
        print(self.track_right_actions)
        
        print()
        print("ACCURACY?:")
        print(self.accuracy*100/(self.frame_per_episode-1))
        
        print("LOCAL ACCURACY?:")
        la=[]
        if self.track_actions[0]==0:
            la.append(0)
        else:
            la.append(self.track_right_actions[0]*100/self.track_actions[0])
            
        if self.track_actions[1]==0:
            la.append(0)
        else:
            la.append(self.track_right_actions[1]*100/self.track_actions[1])
            
        if self.track_actions[2]==0:
            la.append(0)
        else:
            la.append(self.track_right_actions[2]*100/self.track_actions[2])
        print(la)
        
        """
        RETURN RIGHT GUESSES
        """
        acc=self.accuracy*100/(self.frame_per_episode-1)
        if print_stats[0]:
            self.plot_stats(self.models_name_to_dict, self.track_actions, self.track_right_actions, self.frame_per_episode, print_stats[1])
            
        return acc, la
    
    def plot_stats(self, models_names, track_actions, track_right_actions, total_frames, folder_path):
        
        def create_folder_if_not_exists(folder_path):
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
        create_folder_if_not_exists(folder_path)
        # Calcolo delle previsioni corrette totali
        right_guesses = sum(track_right_actions)
        
        # Calcolo della percentuale di predizioni corrette
        accuracy_percentage = (right_guesses * 100) / (total_frames - 1)
        
        # Calcolo della percentuale di accuratezza per ciascun modello
        accuracies = [(right_actions * 100) / total_actions if total_actions != 0 else 0 
                    for right_actions, total_actions in zip(track_right_actions, track_actions)]

        # Grafico 1: Numero di scelte per modello
        plt.figure(figsize=(10, 6))
        plt.bar(models_names, track_actions, color='skyblue')
        plt.title('Number of Actions Chosen per Model')
        plt.ylabel('Number of Actions')
        plt.xlabel('Models')
        
        # Etichette per le barre
        for i, value in enumerate(track_actions):
            plt.text(i, value + 0.5, str(value), ha='center')
        
        plt.savefig(folder_path+'number_of_actions_chosen.png')
        plt.show()

        # Grafico 2: Numero di predizioni corrette per modello
        plt.figure(figsize=(10, 6))
        plt.bar(models_names, track_right_actions, color='lightgreen')
        plt.title('Number of Correct Predictions per Model')
        plt.ylabel('Correct Predictions')
        plt.xlabel('Models')
        
        # Etichette per le barre
        for i, value in enumerate(track_right_actions):
            plt.text(i, value + 0.5, str(value), ha='center')
        
        plt.savefig(folder_path+'correct_predictions_per_model.png')
        plt.show()

        # Grafico 3: Percentuale di predizioni corrette
        plt.figure(figsize=(6, 6))
        plt.pie([right_guesses, total_frames - right_guesses], labels=['Correct', 'Incorrect'],
                autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
        plt.title(f'Correct Predictions Percentage: {accuracy_percentage:.2f}%')
        
        plt.savefig(folder_path+'correct_predictions_percentage.png')
        
        # Grafico 4: Percentuale di accuratezza per ciascun modello
        plt.figure(figsize=(10, 6))
        plt.bar(models_names, accuracies, color='orange')
        plt.title('Accuracy Percentage per Model')
        plt.ylabel('Accuracy (%)')
        plt.xlabel('Models')
        plt.ylim(0, 100)
        
        # Etichette per le barre con la percentuale
        for i, accuracy in enumerate(accuracies):
            plt.text(i, accuracy + 0.5, f'{accuracy:.2f}%', ha='center')

        plt.savefig(folder_path + 'accuracy_percentage_per_model.png')
    
        plt.show()
