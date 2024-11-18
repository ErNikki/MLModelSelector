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

class singleModelTestEnv(Env):
    #def __init__(self,model1,model2,model3):
    def __init__(self, models, test_data_loader, device, frame_per_episode, models_path=Path("../models/")):
        self.actual_ep=0
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
        self.action_space = 2
        
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
        #if not (isinstance(action, int)):
        #    action=action.detach().cpu().item()
        
        
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
            
            case 0:
                if label==predictions[0]:
                    reward=0.3
                    self.accuracy+=1
                    self.track_right_actions[0]+=1
                    
                else:
                    reward=-0.3
            
            case 1:
                if label==predictions[0]:
                    reward=-1
                else:
                    reward=1
                    self.accuracy+=1
                    self.track_right_actions[1]+=1
        
        
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
            #done=True
            #del self.actual_iterable
            self.images, self.labels = next(self.actual_iterable)
            self.actual_image = self.images[self.actual_frame]
            self.actual_label = self.labels[self.actual_frame]
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
    def close(self):
        self.actual_ep+=1
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
        print("GLOBAL ACCURACY?:")
        print(self.accuracy*100/(self.frame_per_episode-1))
        
        print()
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
            
        print(la)
        
        """
        RETURN RIGHT GUESSES
        """
        return self.accuracy/(self.frame_per_episode-1),self.actual_ep
