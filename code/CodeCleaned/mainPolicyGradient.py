import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from pathlib import Path

from torch.distributions import Bernoulli, Categorical
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import gym
import pdb
from utils import ImageClassificationBase, hardware_check, test, to_device

from newTestEnvSimplified import newTestEnvSimplified
from newTrainEnvSimplified import newTrainEnvSimplified

from newTestEnvSimplifiedwithCumulativeRew import newTestEnvSimplifiedCR
from newTrainEnvSimplifiedwithCumulativeRew import newTrainEnvSimplifiedCR

from SingleModelEnvTest import singleModelTestEnv
from SingleModelEnvTrain import singleModelTrainEnv

from Param import *
from SUN397 import Sun397
import random
from tqdm import tqdm
import time



class build_modelV3(ImageClassificationBase):
    def __init__(self, model, output_size):
        super(build_modelV3, self).__init__()
        self.orig_model = model
        self.classify = torch.nn.Linear(1000, output_size)
        self.num_actions=output_size

    def forward(self, x):
        
        x = self.orig_model(x)
        x = self.classify(x)
        x = F.softmax(x,dim=1)
        return x
    
    def act(self, state, device):
        img=torch.FloatTensor(np.float32(state.cpu())).unsqueeze(0).to(device)
        probs = self.forward(img)  # Calcola le probabilità delle 
        # Crea una distribuzione categorica basata sulle probabilità e campiona un'azione
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        
        return action.item()  # Restituisce l'indice dell'azione campionata
        """
        action = torch.argmax(probs, dim=1)  # Trova l'indice dell'azione con la probabilità più alta
        return action
        """


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 1)  # Prob of Left

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-5) # Normalizzazione dei ritorni
    return returns

def plot_stats_train(episode, rewards, losses):
    
    def create_folder_if_not_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
    plt.figure(figsize=(12, 5))

    # Grafico delle ricompense
    plt.subplot(1, 2, 1)
    plt.title(f"Rewards fino all'episodio {episode}")
    plt.plot(rewards, label="Rewards")
    plt.xlabel("Episodi")
    plt.ylabel("Total Rewards")
    plt.grid(True)

    # Grafico delle perdite (loss)
    plt.subplot(1, 2, 2)
    plt.title(f"Loss fino all'episodio {episode}")
    plt.plot(losses, label="Loss", color='red')
    plt.xlabel("Episodi")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.tight_layout()
    # Salva il grafico nel percorso specificato
    plt.savefig(MODEL_STATS_PATH+f"stats_episode.png")
    plt.close()  # Chiude la figura senza mostrarla
    
def plot_stats_test(accuracy_models_current_evolution):
    def create_folder_if_not_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
    path=MODEL_STATS_PATH
    create_folder_if_not_exists(path)

    # Plotting accuracy evolution for current model
    episodes = list(range(1, len(accuracy_models_current_evolution[0]) + 1))
    plt.figure(figsize=(12, 6))

    for model_idx, model_accuracies in enumerate(accuracy_models_current_evolution):
        plt.plot(episodes, model_accuracies, label=f'Model {model_idx + 1}')
        
        # Evidenzia il valore massimo
        max_accuracy = max(model_accuracies)
        max_index = model_accuracies.index(max_accuracy)
        plt.scatter(max_index + 1, max_accuracy, color='red', s=100, zorder=5)  # Marker rosso
        plt.text(max_index + 1, max_accuracy, f'{max_accuracy:.2f}', color='red', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    
    plt.xlabel('Episodes')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Evolution per Model')
    plt.legend()
    plt.savefig(path + 'current_model_accuracy_evolution.png')
    plt.close()
    
def plot_stats_test_model(models_names, track_actions, track_right_actions, total_frames, folder_path):
        
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

def main():

    # Plot duration curve: 
    # From http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    episode_durations = []
    device = hardware_check()
    train=False
    
    TRANSFORM_IMG = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(256),
        #torchvision.transforms.CenterCrop(256),
        torchvision.transforms.Resize((362, 512), antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )
        ])
    
    train_data=Sun397(PATH_TRAIN_FOLDER,TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True,  num_workers = 0)#, prefetch_factor=1 )

    val_data=Sun397(PATH_VAL_FOLDER,TRANSFORM_IMG)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=3606, shuffle=False)#,  num_workers =1, prefetch_factor=1)
    
    """
    current_model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
    current_model = build_modelV3(current_model, 2)
    current_model = to_device(current_model, device)
    """
    
    
    #MODELS_NAME=["MobNetV3S","ResNet18","GoogLeNet"]
    
    MODELS_NAME=["GoogLeNet","ResNet18","MobNetV3S"]  
    print("building test env:")
    test_env=newTestEnvSimplifiedCR(MODELS_NAME, val_data_loader, device, 3606)
    
    if train==True:
            
        print("building main env:")
        train_env=newTrainEnvSimplifiedCR(MODELS_NAME, train_data_loader, device, 32, test_env)
        
        
        """
        print("building test env:")
        test_env=newTestEnvSimplified(MODELS_NAME, val_data_loader, device, 3606)
            
        print("building main env:")
        train_env=newTrainEnvSimplified(MODELS_NAME, train_data_loader, device, 14)
        """
        
        # Parameters
        num_episode = 5000
        warmup = 1
        learning_rate = 0.00001
        gamma = 0.1
        #gamma=0.10
        
        
        current_model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        current_model = build_modelV3(current_model, len(MODELS_NAME))
        current_model = to_device(current_model, device)
        
        #env = gym.make('CartPole-v0')
        policy_net = current_model
        policy_net.train()
        
        
        #optimizer = torch.optim.RMSprop(policy_net.parameters(), lr=learning_rate)
        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)


        # Batch History
        #state_pool = []
        #action_pool = []
        #reward_pool = []
        #steps = 1
        steps=0
        episode_rewards = []
        episode_losses = []
        
        for episode in range(num_episode):
            print(" ")
            print("####################################################################")
            print(f"starting episode {episode}")
            print(f"steps done {steps}")
            rewards = []
            log_probs = []
            state = train_env.reset()
            
            
            # Generazione di un episodio
            while True:
                state = torch.FloatTensor(np.float32(state.cpu())).unsqueeze(0).to(device)
                probs = policy_net(state)
                m = Categorical(probs)
                action = m.sample()
                log_probs.append(m.log_prob(action))
                
                policy_net.eval()
                next_state, reward, done, _ = train_env.step(action.item(), policy_net)
                policy_net.train()
                #next_state, reward, done, _ = train_env.step(action.item())
                rewards.append(reward)
                steps+=1
                
                if done:
                    break
                state = next_state
            
            # Calcolo dei ritorni normalizzati
            returns = compute_returns(rewards, gamma)
            
            # Calcolo della loss e aggiornamento dei pesi
            if steps % warmup == 0:
                policy_losses = []
                for log_prob, R in zip(log_probs, returns):
                    policy_losses.append(log_prob * R)
                    
            
                optimizer.zero_grad()
                policy_loss = torch.cat(policy_losses).mean()
                policy_loss.backward()
                optimizer.step()
            
            
            episode_rewards.append(sum(rewards))
            print("REWARD:" +str(rewards))
            
            episode_losses.append(policy_loss.item())
            print(f"LOSS: {policy_loss.item()}")
            
            
            # Ogni 100 episodi, plotta i risultati
            if episode % 1 == 0 and episode > 0:
                plot_stats_train(episode, episode_rewards, episode_losses)
            #test(test_env, policy_net, 1, device)
      
    else:
        
        def test(env,model,episodes,device, save_stats=False):#(env, model, episodes, render=False, device=device, context=""):
            accuracy_models_current_evolution=[]
            
            print("TESTING EPISODE CURRENT")
            episode_rewards = []
            total_inference_time=0
            steps=0
            for episode in tqdm(range(episodes)):
                state = env.reset()
                episode_reward = 0.0
                while True:
                    with torch.no_grad():
                        
                        torch.cuda.synchronize()
                        start_time = time.time()  # Inizio del conteggio
                        action = model.act(state, device)
                        torch.cuda.synchronize()
                        inference_time_external = time.time() - start_time  # Fine del conteggio
                        
                    next_state, reward, done, inference_time_internal = env.step(action)

                    #inutile
                    episode_reward += reward
                    state = next_state
                    steps+=1
                    
                    if done:
                        break
                    total_inference_time+= (inference_time_external+inference_time_internal)
            
            right_guesses, models_accuracies, models_name_to_dict, track_actions, track_right_actions, frame_per_episode = env.close()
            
            #print(f"Steps: {steps}")
            print()
            print("INFERENCE")
            print(f"TOTAL TIME IN (ms): {total_inference_time*1000}, mean (ms): {total_inference_time*1000/steps}")
            
            if(save_stats):
                print("saving stats")
                path=MODEL_STATS_TEST_PATH
                plot_stats_test(models_name_to_dict, track_actions, track_right_actions, frame_per_episode, path)
            
            return right_guesses
        
        
        current_model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        current_model = build_modelV3(current_model, len(MODELS_NAME))
        current_model = to_device(current_model, device)
        
        policy_net = current_model
        #policy_net.eval()
        
        TARGET_MODEL_FILE="target_net"
        MODEL_NAME=TARGET_MODEL_FILE+".pth"
        env_path = Path(MODEL_SAVE_PATH)
        pth_path_model = Path(env_path, MODEL_NAME)
        
        policy_net.load_state_dict(torch.load(pth_path_model))
        policy_net.eval()
        
        cumulative_reward_list=test(test_env, policy_net, 1, device, False)
        
    
if __name__ == '__main__':
    main()