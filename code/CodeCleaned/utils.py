from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
#from torchsummaryX import summary
from torchsummary import summary
import torchvision
from torchvision.transforms import v2
import random
import numpy as np

import os
import sys
import gym
from Param import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_stats(frame_idx, rewards, losses, acc1, acc2, accuracy_models_current_evolution, accuracy_models_target_evolution):
    def create_folder_if_not_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    create_folder_if_not_exists(MODEL_STATS_PATH)
    clear_output(True)

    #plt.figure(figsize=(30, 15))
    plt.figure(figsize=(19, 10))

    # Rewards subplot
    "REWARDS"
    plt.subplot(121)
    #plt.title(f'Total frames {frame_idx}. Avg reward over last 10 episodes: {np.mean(rewards[-10:])}')
    #plt.plot(rewards)
    
    #Creazione del grafico
    plt.plot(rewards, color='blue', alpha=0.7)  # Disegna la curva delle ricompense
    plt.title('Rewards Over Time', fontsize=14)  # Titolo del grafico
    plt.xlabel('Episodes', fontsize=12)  # Etichetta asse x
    plt.ylabel('Reward', fontsize=12)  # Etichetta asse y
    plt.grid(True)  # Abilita la griglia per una visualizzazione migliore
    # Aggiunta della media mobile per una visione più fluida delle ricompense
    if frame_idx//1024>10:
        window = len(rewards)-10  # Imposta la finestra della media mobile
    else:
        window=1
    rewards_smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, len(rewards)), rewards_smoothed, color='red', linewidth=2, label=f'Moving Avg (window={window})')
    

    # Losses subplot
    "LOSS"
    plt.subplot(122)
    #plt.title('Loss')
    #plt.plot(losses)
    # Imposta la dimensione della figura
    plt.plot(losses, color='blue', alpha=0.7)  # Disegna la curva della loss
    plt.title('Loss Over Time', fontsize=14)  # Titolo del grafico
    plt.xlabel('Images', fontsize=12)  # Etichetta asse x
    plt.ylabel('Loss', fontsize=12)  # Etichetta asse y
    plt.grid(True)  # Abilita la griglia per una visualizzazione migliore

    # Aggiunta della media mobile per una visione più fluida delle loss
    if frame_idx//1024>50:
        window = len(losses)-50  # Imposta la finestra della media mobile
    else:
        window=1
    losses_smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(window-1, len(losses)), losses_smoothed, color='red', linewidth=2, label=f'Moving Avg (window={window})')

    plt.savefig(MODEL_STATS_PATH + "loss_reward_stats.png")
    plt.close()
    
    
    #plt.figure(figsize=(30, 15))
    plt.figure(figsize=(19, 10))
    
    # Accuracy Current Model subplot
    "ACC CURRENT"
    plt.subplot(121)
    plt.title('Accuracy Current Model')
    plt.plot(acc1)

    # Evidenziare la massima accuracy per il modello corrente
    max_acc1 = max(acc1)
    max_idx1 = acc1.index(max_acc1)
    plt.scatter(max_idx1, max_acc1, color='red', s=100, label='Max Accuracy', zorder=5)
    plt.text(max_idx1, max_acc1, f'{max_acc1:.2f}', color='red', fontsize=12, verticalalignment='bottom', horizontalalignment='right')


    # Accuracy Target Model subplot
    "ACC TARGET"
    plt.subplot(122)
    plt.title('Accuracy Target Model')
    plt.plot(acc2)

    # Evidenziare la massima accuracy per il modello target
    max_acc2 = max(acc2)
    max_idx2 = acc2.index(max_acc2)
    plt.scatter(max_idx2, max_acc2, color='red', s=100, label='Max Accuracy', zorder=5)
    plt.text(max_idx2, max_acc2, f'{max_acc2:.2f}', color='red', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    plt.legend()

    # Salvare la figura con le accuracy di entrambi i modelli
    plt.savefig(MODEL_STATS_PATH + "accuracies_target_current.png")
    plt.close()

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
    plt.savefig(MODEL_STATS_PATH + 'current_model_accuracy_evolution.png')
    plt.close()

    # Plotting accuracy evolution for target model
    plt.figure(figsize=(12, 6))

    for model_idx, model_accuracies in enumerate(accuracy_models_target_evolution):
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
    plt.savefig(MODEL_STATS_PATH + 'target_model_accuracy_evolution.png')
    plt.close()




def compute_loss(model, replay_buffer, batch_size, gamma, device=device):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    q_values_old = model(state)
    q_values_new = model(next_state)

    q_value_old = q_values_old.gather(1, action.unsqueeze(1)).squeeze(1)
    q_value_new = q_values_new.max(1)[0]
    expected_q_value = reward + gamma * q_value_new * (1 - done)

    loss = (q_value_old - expected_q_value.data).pow(2).mean()

    return loss

def compute_loss_double(current_model,target_model, replay_buffer, batch_size, gamma, device=device):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(np.float32(state)).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    done = torch.FloatTensor(done).to(device)

    q_values_old = current_model(state)
    q_values_new = current_model(next_state)
    target_q_values_new = target_model(next_state)

    q_value_old = q_values_old.gather(1, action.unsqueeze(1)).squeeze(1)
    q_value_new = target_q_values_new.gather(1, torch.max(q_values_new, 1)[1].unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * q_value_new * (1 - done)

    loss = (q_value_old - expected_q_value.data).pow(2).mean()

    return loss

def train_doublev2(env, test_env, current_model, target_model, optimizer, replay_buffer, device=device):
    steps_done = 0
    episode_rewards = []
    losses = []
    accuracy_current_model=[]
    accuracy_target_model=[]
    accuracy_models_current_evolution=[]
    accuracy_models_target_evolution=[]
    current_model.train()
    target_model.train()
    best_reward_curr_model=0
    best_reward_target_model=0
    print_stats=(False,"")
    """
    Da decommentare
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=REDUCE_LR_STEPS, gamma=0.001)"""
    if not os.path.exists(MODEL_SAVE_PATH):
            os.makedirs(MODEL_SAVE_PATH)

    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0.0
        print("##############################################################################################")
        print(f"STARTING ESPIDOSE {(episode)}")
        print(" ")
        while True:
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(- steps_done / EPS_DECAY)
            
            #state=torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1.transforms(state)
            #augmenter = v2.AugMix(state)
            
            action = current_model.act(state, epsilon, device)
            steps_done += 1
            
            next_state, reward, done, _ = env.step(action)
            """
            for simplifiedEnvwithCR
            """
            #next_state, reward, done, _ = env.step(action, current_model)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(replay_buffer) > INITIAL_MEMORY :
                
                loss = compute_loss_double(current_model, target_model, replay_buffer, BATCH_SIZE, GAMMA, device)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                

            if steps_done % UPDATE_TARGET == 0 and steps_done > WARMUP:
                target_model.load_state_dict(current_model.state_dict())

            #if steps_done % PLOT_STATS == 0 and steps_done > WARMUP:
                #plot_stats(steps_done, episode_rewards, losses)
                
            if done:
                print(f"REWARD: {episode_reward}")
                episode_rewards.append(episode_reward)
                break
            
            
        if (episode+1) % 1 == 0 and steps_done > WARMUP:
            print(" ")
            print("####")
            print("TESTING EPISODE CURRENT")
            current_model.eval()
            result, models_accuracies=test(test_env, current_model, 1, device, print_stats)
            
            if (not accuracy_models_current_evolution) and len(models_accuracies)>0:
                accuracy_models_current_evolution = [[] for _ in models_accuracies]
                
            accuracy_models_current_evolution= [a + [b] for a, b in zip(accuracy_models_current_evolution, models_accuracies)]
            
            accuracy_current_model.append(result)
            
            if result>best_reward_curr_model:
                best_reward_curr_model=result
                curr_path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_curr_episode_{episode+1}.pth")
                print(f"Saving curr weights at Episode {episode+1} ...")
                torch.save(current_model.state_dict(), curr_path)
            current_model.train()
            
            print(" ")
            print("###")
            print("TESTING EPISODE TARGET")
            target_model.eval()
            
            result, models_accuracies=test(test_env, target_model, 1, device, print_stats)
            
            if (not accuracy_models_target_evolution) and len(models_accuracies)>0:
                accuracy_models_target_evolution = [[] for _ in models_accuracies]
                
            accuracy_models_target_evolution= [a + [b] for a, b in zip(accuracy_models_target_evolution, models_accuracies)]
            accuracy_target_model.append(result)
            if result>best_reward_target_model:
                best_reward_target_model=result
                target_path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_target_episode_{episode+1}.pth")
                print(f"Saving target weights at Episode {episode+1} ...")
                torch.save(target_model.state_dict(), target_path)
                
            target_model.train()
        """
        commento solo momentaneo
        
        if(steps_done>WARMUP):
            scheduler.step()
        """
        
        
        if (episode+1) % 1 == 0 and episode+1 >= 2 and steps_done > WARMUP:
            print(" ")
            print("SAVING STATS")
            plot_stats(steps_done, episode_rewards, losses, accuracy_current_model, accuracy_target_model, accuracy_models_current_evolution, accuracy_models_target_evolution)  
        #sys.stdout.flush()
        
        print("##############################################################################################")
    env.close()

def test(env,model,episodes,device, print_stats):#(env, model, episodes, render=False, device=device, context=""):
    #model.eval()
    episode_rewards = []
    total_inference_time=0
    for episode in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0.0
        steps=0
        while True:
            with torch.no_grad():
                
                start_time_external = time.time()  # Inizio del conteggio
                action = model.act(state, 0, device)
                inference_time_external = time.time() - start_time_external  # Fine del conteggio
                
            next_state, reward, done, inference_time_internal = env.step(action)
            steps+=1
            #inutile
            episode_reward += reward
            state = next_state
            
            if done:
                break
            
            
            total_inference_time+= (inference_time_external+inference_time_internal)
    
    right_guesses, models_accuracies=env.close(print_stats)
    print(f"TOTAL TIME: {total_inference_time*1000}, mean: {total_inference_time*1000/steps}")
    return right_guesses, models_accuracies

def hardware_check():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("Actual device: ", device)
    if 'cuda' in device:
        print("Device info: {}".format(str(torch.cuda.get_device_properties(device)).split("(")[1])[:-1])

    return device

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def load_model(model_name, pth_path, device):
        model = None

        if model_name == "Efficientnet_b0":
            model = to_device(
                build_model(
                    model=torchvision.models.efficientnet_b0(
                        weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "Efficientnet_b1":
            model = to_device(
                build_model(
                    model=torchvision.models.efficientnet_b1(
                        weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "Efficientnet_b3":
            model = to_device(
                build_model(
                    model=torchvision.models.efficientnet_b3(
                        weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "GoogLeNet":
            model = to_device(
                build_model(
                    model=torchvision.models.googlenet(
                        weights=torchvision.models.GoogLeNet_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "Mnasnet0_5":
            model = to_device(
                build_model(
                    model=torchvision.models.mnasnet0_5(
                        weights=torchvision.models.MNASNet0_5_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "Mnasnet0_75":
            model = to_device(
                build_model(
                    model=torchvision.models.mnasnet0_75(
                        weights=torchvision.models.MNASNet0_75_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "Mnasnet0_10":
            model = to_device(
                build_model(
                    model=torchvision.models.mnasnet1_0(
                        weights=torchvision.models.MNASNet1_0_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "MobNetV3S":
            model = to_device(
                build_model(
                    model=torchvision.models.mobilenet_v3_small(
                        weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "MobNetV3L":
            model = to_device(
                build_model(
                    model=torchvision.models.mobilenet_v3_large(
                        weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "RegNet_x_400mf":
            model = to_device(
                build_model(
                    model=torchvision.models.regnet_x_400mf(
                        weights=torchvision.models.RegNet_X_400MF_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "RegNet_x_800mf":
            model = to_device(
                build_model(
                    model=torchvision.models.regnet_x_800mf(
                        weights=torchvision.models.RegNet_X_800MF_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "RegNet_y_400mf":
            model = to_device(
                build_model(
                    model=torchvision.models.regnet_y_400mf(
                        weights=torchvision.models.RegNet_Y_400MF_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "RegNet_y_800mf":
            model = to_device(
                build_model(
                    model=torchvision.models.regnet_y_800mf(
                        weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "ResNet18":
            model = to_device(
                build_model(
                    model=torchvision.models.resnet18(
                        weights=torchvision.models.ResNet18_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "ShufflNetV2_x05":
            model = to_device(
                build_model(
                    model=torchvision.models.shufflenet_v2_x0_5(
                        weights=torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "ShufflNetV2_x10":
            model = to_device(
                build_model(
                    model=torchvision.models.shufflenet_v2_x1_0(
                        weights=torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "ShufflNetV2_x15":
            model = to_device(
                build_model(
                    model=torchvision.models.shufflenet_v2_x1_5(
                        weights=torchvision.models.ShuffleNet_V2_X1_5_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "ShufflNetV2_x20":
            model = to_device(
                build_model(
                    model=torchvision.models.shufflenet_v2_x2_0(
                        weights=torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "Squeezenet1_0":
            model = to_device(
                build_model(
                    model=torchvision.models.squeezenet1_0(
                        weights=torchvision.models.SqueezeNet1_0_Weights.DEFAULT
                    )
                ),
                device,
            )
        elif model_name == "Squeezenet1_1":
            model = to_device(
                build_model(
                    model=torchvision.models.squeezenet1_1(
                        weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT
                    )
                ),
                device,
            )
        else:
            raise NotImplementedError(f"{model_name} has not been implemented yet!")
        
        #elif model_to_load is None:
            #Log.mfatal(ModelLoader._MODULE, "Could not load model")

        #model_loaded = to_device(build_model(model_to_load), self._device)
        
        model.load_state_dict(torch.load(pth_path))

        # set to eval mode
        model.eval()

        # freeze layers
        for parameter in model.parameters():
            parameter.requires_grad = False

        return model
    
class ImageClassificationBase(torch.nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = torch.nn.functional.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = torch.nn.functional.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        return result['val_loss'], result['val_acc']
    
@torch.no_grad()
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

class build_model(ImageClassificationBase):
    def __init__(self, model):
        super(build_model, self).__init__()
        self.orig_model = model
        self.classify = torch.nn.Linear(1000, 271)

    def forward(self, x):
        x = self.orig_model(x)
        x = self.classify(x)
        return x


class build_modelV2(ImageClassificationBase):
    def __init__(self, model, output_size):
        super(build_modelV2, self).__init__()
        self.orig_model = model
        self.classify = torch.nn.Linear(1000, output_size)
        self.num_actions=output_size

    def forward(self, x):
        x = self.orig_model(x)
        x = self.classify(x)
        return x

    def act(self, state, epsilon, device):
        if random.random() >= epsilon:
            state = torch.FloatTensor(np.float32(state.cpu())).unsqueeze(0).to(device)
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        return action
