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

#from PIL import Image
#from collections import Counter

#num of frame per episode forse dopo sarebbe meglio farne una randomica con tot episodi
#FRAME_PER_EPISODE=512

#APPLICARE unsqueeze(1) IN RETURN COSI DA AVERE 1X84X84 RICORDA CHE INPUT DI CONV2D è IL NUMERO DEI CANALI IN INPUT!! VA LEVATO SE POI SI DECIDE DI NON USARE 84X84

class newTrainEnvSimplified(Env):
    #def __init__(self,model1,model2,model3):
    def __init__(self, models, train_data_loader, device, frame_per_episode, models_path=Path("../models/")):
        
        #Used inside train to specify name of file
        self.spec_id="netSelector"
        
        #device used to process images
        self.device = device
        
        #save dataloader in order to re-shuffle it at the end of all batches
        self.train_data_loader=train_data_loader    
        #build iterable to iterate on dataloader
        
        self.actual_iterable=iter(train_data_loader)
        
        #used to stop episode after tot framed processed
        self.frame_per_episode=frame_per_episode
        #count of many frames are processed during episode in order to end it if overcome frame_per_episode 
        self.actual_frame=0
        
        
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
        
        
        self.models_name=models
        #index=0
        for model_name in models:
            pth_path_model = Path(models_path, model_name)
            pth_path_model = os.path.join(pth_path_model, "best_valAcc_model.pth")
            print(pth_path_model)
            model= load_model(model_name, pth_path_model, self.device)
            self.models_loaded.append(model)
            #self.track_actions.append(0)
            #self.models_name_to_dict.append(model_name)
            #self.track_right_actions_per_model.append(0)
            #index+=1
        #print(self.models_loaded)
        #self.augmenter = v2.AugMix()
    
    def step(self, action):     
        
        """
        unsqueeze
        """
        #FROM [3,362,512] TO [1,3,362,512] in order to process the image using agent model
        
        #augmented_img=self.augmenter(self.actual_image)
        img = self.actual_image.unsqueeze(0).to(self.device)
        
        label = self.actual_label
        
        
        """
        image processing
        """
        #Usually the action choosen random is an int, the action choosen by the model is a tensor
        if not (isinstance(action, int)):
            action=action.detach().cpu().item()
        
        """
        GET PREDICTIONS OF MODELS
        """
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
        
        match action:
            
            #primo modello unico
            case 0:
                
                if predictions[0]==label:
                    reward=0.3
                else:
                    reward=-1
                
            case 1:
                
                if label==predictions[1]:
                    if predictions.count(label)==1:
                        reward=1
                    else:
                        reward=0.2
                else:
                    reward=-1
                    
            case 2:
               
                if label==predictions[2]:
                    if predictions.count(label)==1:
                        reward=1
                    else:
                        reward=0.2
                    
                else:
                    reward=-1

                
                
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
        
        self.episode_reward=0
        self.actual_frame=0
                
        #read netx bash of images
        del self.images 
        del self.labels
        
        try:
            self.images, self.labels = next(self.actual_iterable)
            
        except:
            del self.actual_iterable
            self.actual_iterable=iter(self.train_data_loader)
            self.images, self.labels = next(self.actual_iterable)
        
        #get first image "start state"
        self.actual_image = self.images[self.actual_frame]
        self.actual_label = self.labels[self.actual_frame]

        return self.actual_image
        
    def close(self):

        pass
        