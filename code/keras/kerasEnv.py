import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.applications
from keras.models import Model, load_model



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

class _NetSelectorEnv(Env):
    #def __init__(self,model1,model2,model3):
    def __init__(self, models, image_size, train_data_loader, device, frame_per_episode, train_mode=True, models_path=Path("./models/")):
        
        #Used inside train to specify name of file
        self.spec_id="netSelector"
        
        #device used to process images
        self.device = device
        
        """
        Stats variables
        """
        #track actions
        self.track_actions=[]
        #track right guesses of models 
        self.track_right_actions_per_model=[]
        #track models name associated with index to better display
        self.models_name_to_dict=[]
        
        """
        actually NOT used
        """
        #used to elaborate custom reward
        self.previous_reward=0 
        
        """
        actually NOT used in the actual train cause the shuffle start at then end of dataloader iteration
        """
        #FLAG utilizzato nel train per creare un nuovo iter e quindi fare shuffle
        self.train_mode=train_mode
        
        #save dataloader in order to re-shuffle it at the end of all batches
        self.train_data_loader=train_data_loader    
        #build iterable to iterate on dataloader
        if not self.train_mode:
            self.actual_iterable=cycle(iter(train_data_loader))
        else:
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
        self.action_space = Discrete(len(models))
        
        #used to give about the dimension of the images processed by env
        #self.observation_space = Box(0, 255, (1, 84, 84), np.uint8)
        self.observation_space = Box(0,255,image_size , np.uint8)

        """
        LOADING MODELS
        """
        
        self.models_loaded=[]
        
        #DA USARE COME VARIABILE INIT
        
        
        self.models_name=models
        #index=0
        for model_name in models:
            #pth_path_model = Path(env_path, model_name)
            #pth_path_model = os.path.join(pth_path_model, "best_valLoss_model.pth")
            path_model=models_path+"/"+model_name+"/best_loss.h5"
            print(path_model)
            model= load_model(path_model)
            self.models_loaded.append(model)
            self.track_actions.append(0)
            self.models_name_to_dict.append(model_name)
            self.track_right_actions_per_model.append(0)
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
        
        
        #based on the number of the action choose a model
        model=self.models_loaded[action]
        #process image in order to get predicted label by the chosen model
        with torch.no_grad():
            pred = model(img)
            pred = pred.detach().cpu().numpy()
            predicted_label=np.argmax(pred)
            
        #stats to track action taken in a single episode by every model
        self.track_actions[action]+=1       
        
        """
        computing reward
        """
        if(label==predicted_label):
            if action==0:
                reward=1
                
            elif action==1:
                reward=0.7
                
            elif action==2:
                reward=0.7
                
                
            self.episode_reward+=reward
            #stats to track RIGHT action taken in a single episode by every model
            self.track_right_actions_per_model[action]+=1
            
            #TO USE CUSTOM REWARD
            #reward=self.reward_function(1,self.previous_reward)
            
        else:
            reward=-1
            
            #TO USE CUSTOM REWARD
            #reward=self.reward_function(0,self.previous_reward)
        
        #TO USE CUSTOM REWARD
        #self.previous_reward=reward
        #self.episode_reward+=reward
        
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
            self.actual_iterable=iter(self.train_data_loader)
            return self.actual_image, reward, done, 0
        

        """
        differents try to resize images
        """
        #RESIZE IMAGE TO 84X84 THEN RESCALE USING APPROXIMATION https://e2eml.school/convert_rgb_to_grayscale
        #l'immagine gestita dal modello ha una size 1x84x84
        #img_transformed=(v2.Resize(size=(84,84))(self.actual_image))
        #img_transformed=self.rgb2gray_approx(img_transformed)

        #lo 0 è un refuso che poi va eliminato ovunque        
        
        #caso 3x362x512
        #return self.actual_image, reward, done, 0

        #caso 84x84 (grayscale)
        #return img_transformed.unsqueeze(0), reward, done, 0

        #caso 3x84x84
        #return img_transformed, reward, done, 0
            
                
                
            

    def render(self):
    # not needed 
        pass
    
    def reset(self):
        
        #setup episode reward counter and actual frame (state) counter of the environment
        self.episode_reward=0
        self.actual_frame=0
        
        #se siamo in train mode creamo un nuovo iterable in modo da fare shuffle del dataloader
        #OLD training
        
        """
        if self.train_mode:
            self.actual_iterable=iter(self.train_data_loader)
        """
        if not self.train_mode:
            #self.actual_iterable=iter(self.train_data_loader)
            self.track_actions=[]
            self.models_name_to_dict=[]
            self.track_right_actions_per_model=[]     
            for model_name in self.models_name:
                self.track_actions.append(0)
                self.models_name_to_dict.append(model_name)
                self.track_right_actions_per_model.append(0)            
           
            
        #read netx bash of images
        self.images, self.labels = next(self.actual_iterable)
        
        #get first image "start state"
        self.actual_image = self.images[self.actual_frame]
        self.actual_label = self.labels[self.actual_frame]

        """
        trying resizing
        """
        #RESIZE IMAGE TO 84X84 THEN RESCALE USING APPROXIMATION https://e2eml.school/convert_rgb_to_grayscale
        #l'immagine gestita dal modello ha una size 1x84x84
        #img_transformed=(v2.Resize(size=(84,84))(self.actual_image))
        #img_transformed=self.rgb2gray_approx(img_transformed)
        
        #caso 3x362x512
        return self.actual_image
        
        #caso 84x84 grayscale
        #return img_transformed.unsqueeze(0)

        #caso 3x84x84 senza grayscale
        #return img_transformed
    
    #usato nel train ma qui inutile
    def close(self):

        if not self.train_mode:
            right_guesses=0
            for index in range(len(self.models_name_to_dict)):
                print(str(self.models_name_to_dict[index])+" has been chosen: "+ str(self.track_actions[index]))
                print(str(self.models_name_to_dict[index])+ "right guesses: " + str(self.track_right_actions_per_model[index]))
                print(" ")
                right_guesses+=self.track_right_actions_per_model[index]
            print("Total Frames: " + str(self.frame_per_episode))
            print("Correct predictions: " + str(right_guesses))
            print("Percentage: " + str(right_guesses*100/self.frame_per_episode))
            return right_guesses

        else:
            pass
        

    def rgb2gray_approx(self,rgb_img):
        """
        Convert *linear* RGB values to *linear* grayscale values.
        """
        red = rgb_img[0, :, :]
        green = rgb_img[1, :, :]
        blue = rgb_img[2, :, :]
    
        gray_img = (
            0.299 * red
            + 0.587 * green
            + 0.114 * blue)
    
        return gray_img
    
    def reward_function(self,accuracy, previous_accuracy, exploration_penalty=0.1):
        """
        Calculates the reward based on the accuracy of the selected model and its change from previous selections.
        
        Parameters:
            accuracy (float): Accuracy of the current selected model.
            previous_accuracy (float): Accuracy of the previously selected model.
            exploration_penalty (float): Penalty for exploration to encourage exploitation.
        
        Returns:
            reward (float): The reward for selecting the current model.
        """
        improvement = accuracy - abs(previous_accuracy)
        
        # Reward based on accuracy
        reward = accuracy
        
        # Penalize exploration to encourage exploitation
        reward -= exploration_penalty
        
        # Bonus for improvement from previous selection
        if improvement > 0:
            reward += improvement * 0.5  # Weighted bonus for improvement
        
        # Penalty for decrease in accuracy from previous selection
        else:
            reward -= abs(improvement) * 0.5  # Weighted penalty for decrease
        
        return reward
    