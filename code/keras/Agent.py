import random
import gym
import numpy as np
from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from tqdm import tqdm
import os
import torchvision
import torch

import sys
sys.path.append('../')

from Environment import _NetSelectorEnv
from SUN397 import Sun397
from utils import hardware_check




class DDQN_Agent:
    #
    # Initializes attributes and constructs CNN model and target_model
    #
    def __init__(self, state_size, action_size, model=None, target_model=None):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        
        # Hyperparameters
        self.gamma = 0.1           # Discount rate
        self.epsilon = 0          # Exploration rate
        self.epsilon_min = 0      # Minimal exploration rate (epsilon-greedy)
        self.epsilon_decay = 0  # Decay rate for epsilon
        self.update_rate = 512    # Number of steps until updating the target network
        
        # Construct DQN models
        if model==None:
            self.model = self._build_model()
            self.target_model = self._build_model()
            self.target_model.set_weights(self.model.get_weights())
            self.model.summary()
        else:
            self.model = model
            self.target_model=target_model

    #
    # Constructs CNN
    #
    def _build_model(self):
        model = Sequential()
        
        # Conv Layers
        model.add(Conv2D(32, (8, 8), strides=4, padding='same', input_shape=self.state_size))
        model.add(Activation('relu'))
        
        model.add(Conv2D(64, (4, 4), strides=2, padding='same'))
        model.add(Activation('relu'))
        
        model.add(Conv2D(64, (3, 3), strides=1, padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())

        # FC Layers
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=keras.optimizers.RMSprop( learning_rate=0.00025, rho=0.95, epsilon=None, decay=0.0)
)
        return model

    #
    # Stores experience in replay memory
    #
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    #
    # Chooses action based on epsilon-greedy policy
    #
    def act(self, state):
        # Random exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(state)
        
        return np.argmax(act_values[0])  # Returns action using policy

    #
    # Trains the model using randomly selected experiences in the replay memory
    #
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            
            if not done:
                max_action = np.argmax(self.model.predict(next_state)[0])
                target = (reward + self.gamma * self.target_model.predict(next_state)[0][max_action])
            else:
                target = reward
                
            # Construct the target vector as follows:
            # 1. Use the current model to output the Q-value predictions
            target_f = self.model.predict(state)
            
            # 2. Rewrite the chosen action value with the computed target
            target_f[0][action] = target
            
            # 3. Use vectors in the objective computation
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #
    # Sets the target model parameters to the current model parameters
    #
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
            
    #
    # Loads a saved model
    #
    def load(self, name):
        self.model.load_weights(name)

    #
    # Saves parameters of a trained model
    #
    def save(self, name):
        self.model.save_weights(name)
        
        
def test(env,model,episodes,device):#(env, model, episodes, render=False, device=device, context=""):
    #model.eval()
    episode_rewards = []
    for episode in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0.0
        while True:
            
            act_values = self.model.predict(state)
            action=np.argmax(act_values[0])
                
            next_state, reward, done, _ = env.step(action)

            #if render:
            #    env.render()
            #    time.sleep(0.02)

            episode_reward += reward
            state = next_state
            
            if done:
                #episode_rewards.append(episode_reward)
                #print(f"Finished Episode {episode+1} with reward {episode_reward}")
                break

            
if __name__ == "__main__":
    
    train_mode=False
    #double_train=True
    #x2_train=True
    device = hardware_check()
    PATH_DATASET_FOLDER = os.getcwd() + "/../dataset/SUN397/"
    PATH_TRAIN_FOLDER=PATH_DATASET_FOLDER+  "train_selector"
    PATH_TEST_FOLDER=PATH_DATASET_FOLDER+"/test"
    PATH_VAL_FOLDER=PATH_DATASET_FOLDER+"/val_selector"
    PATH_VAL_MODELS_FOLDER=PATH_DATASET_FOLDER+"/val_models"
    
    
    TRANSFORM_IMG = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(256),
        #torchvision.transforms.CenterCrop(256),
        torchvision.transforms.Resize((362, 512), antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )
        ])
    
    train_data=Sun397(PATH_TRAIN_FOLDER, TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1024, shuffle=True,  num_workers = 0)#, prefetch_factor=1 )

    val_data=Sun397(PATH_VAL_FOLDER, TRANSFORM_IMG)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=3606, shuffle=False)#,  num_workers =1, prefetch_factor=1)
    
    MODELS_NAME=["MobNetV3S","ResNet18","ShufflNetV2_x05"]
    print("#### Building envs ####")
    
    print("building main env:")
    env=_NetSelectorEnv(MODELS_NAME, train_data_loader, device, 1024, train_mode)
    print("building test env:")
    train_env=_NetSelectorEnv(MODELS_NAME, val_data_loader, device, 3606, train_mode= False)
    """
    model = keras.applications.Xception(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=3,
        classifier_activation="softmax",
    )
    
    target_model=keras.applications.Xception(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=3,
        classifier_activation="softmax",
    )
    """
    
    state_size = (3,362,512)
    action_size = 3
    agent = DDQN_Agent(state_size, action_size)
    #agent.load('models/')

    episodes = 1000
    batch_size = 16
    
    #skip_start = 90  # MsPacman-v0 waits for 90 actions before the episode begins
    #total_time = 0   # Counter for total number of steps taken
    #all_rewards = 0  # Used to compute avg reward over time
    #blend = 4        # Number of images to blend
    done = False
    warmup=10024
    
    for e in range(episodes):
        steps_done = 0
        episode_rewards = []
        #losses = []
        #current_model.train()
        total_reward = 0
        game_score = 0
        state = env.reset()
        #images = deque(maxlen=blend)  # Array of images to be blended
        #images.append(state)
        
        #for skip in range(skip_start): # skip the start of each game
        #    env.step(0)
        
        while True:
            #env.render()
            #total_time += 1
            steps_done += 1
            
            # Every update_rate timesteps we update the target network parameters
            if steps_done % agent.update_rate == 0 and steps_done > warmup:
                agent.update_target_model()
            
            # Return the avg of the last 4 frames
            #state = blend_images(images, blend)
            
            # Transition Dynamics
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            
            #game_score += reward
            #total_reward += reward
            
            # Return the avg of the last 4 frames
            
            #next_state = process_frame(next_state)
            #images.append(next_state)
            #next_state = blend_images(images, blend)
            
            # Store sequence in replay memory
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
                
            if len(agent.memory) > batch_size and steps_done > warmup:
                agent.replay(batch_size)
                
            if done:
                print("TESTING EPISODE CURRENT")
                result=test(test_env, agent.model, 1, device)
                """
                if result>best_reward_curr_model:
                    best_reward_curr_model=result
                    curr_path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_curr_episode_{episode+1}.pth")
                    print(f"Saving curr weights at Episode {episode+1} ...")
                    torch.save(current_model.state_dict(), curr_path)
                    
                episode_rewards.append(result)
                print("TESTING EPISODE TARGET")
                """
                
                result=test(test_env, agent.target_model, 1, device)
                """
                if result>best_reward_target_model:
                    best_reward_target_model=result
                    target_path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_target_episode_{episode+1}.pth")
                    print(f"Saving target weights at Episode {episode+1} ...")
                    torch.save(target_model.state_dict(), target_path)
                """
                #all_rewards += game_score
                
                #print("episode: {}/{}, game score: {}, reward: {}, avg reward: {}, time: {}, total time: {}"
                #    .format(e+1, episodes, game_score, total_reward, all_rewards/(e+1), time, total_time))
                
                break