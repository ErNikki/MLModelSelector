import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from DistributionalDQN import distr_projection

import os

#import gym
import matplotlib.pyplot as plt

from tqdm import tqdm

from ParamDDQN import *

from IPython.display import clear_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_double(env, current_model,target_model, optimizer, replay_buffer, device=device):
    steps_done = 0
    episode_rewards = []
    losses = []
    current_model.train()
    for episode in tqdm(range(EPISODES)):
        state = env.reset()
        episode_reward = 0.0
        while True:
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(- steps_done / EPS_DECAY)
            action = current_model.act(state, epsilon, device)
            #print(action)
            #print(" ")
            steps_done += 1

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(replay_buffer) > INITIAL_MEMORY:
                loss = calc_loss(current_model, target_model, replay_buffer, BATCH_SIZE, GAMMA, device)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if steps_done % 256 == 0:
                target_model.load_state_dict(current_model.state_dict())

            if steps_done % 1024 == 0:
                plot_stats(steps_done, episode_rewards, losses)

            if done:
                episode_rewards.append(episode_reward)
                break
            
        if (episode+1) % 100 == 0:
            curr_path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_curr_episode_{episode+1}.pth")
            target_path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_target_episode_{episode+1}.pth")
            print(f"Saving weights at Episode {episode+1} ...")
            torch.save(current_model.state_dict(), curr_path)
            torch.save(target_model.state_dict(), target_path)
    env.close()
    
def train_doublev2(env, test_env, current_model, target_model, optimizer, replay_buffer, device=device):
    steps_done = 0
    episode_rewards = []
    losses = []
    current_model.train()
    best_reward_curr_model=0
    best_reward_target_model=0
    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0.0
        print(f"STARTING ESPIDOSE {(episode)}")
        while True:
            #epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(- steps_done / EPS_DECAY)
            action = current_model.act(state, 1, device)
            steps_done += 1

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(replay_buffer) > INITIAL_MEMORY:
                
                loss = calc_loss(current_model, target_model, replay_buffer, BATCH_SIZE, GAMMA, device)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if steps_done % 2048 == 0:
                target_model.load_state_dict(current_model.state_dict())

            #if steps_done % 1024 == 0:
            #    plot_stats(steps_done, episode_rewards, losses)

            if done:
                episode_rewards.append(episode_reward)
                break
            
        if (episode+1) % 1 == 0:
            print("TESTING EPISODE")
            result=test(test_env, current_model, 1, device)
            if result>best_reward_curr_model:
                best_reward_curr_model=result
                curr_path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_curr_episode_{episode+1}.pth")
                print(f"Saving curr weights at Episode {episode+1} ...")
                torch.save(current_model.state_dict(), curr_path)
            result=test(test_env, target_model, 1, device)
            if result>best_reward_target_model:
                best_reward_target_model=result
                target_path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_target_episode_{episode+1}.pth")
                print(f"Saving target weights at Episode {episode+1} ...")
                torch.save(target_model.state_dict(), target_path)
            
    env.close()

def calc_loss(net, tgt_net, replay_buffer, batch_size, gamma, device="cpu"):
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    #batch_size = len(batch)
    

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    #rewards = torch.FloatTensor(rewards)
    #dones = torch.ByteTensor(dones).to(device)

    # next state distribution
    next_distr_v, next_qvals_v = tgt_net.both(next_states_v)
    next_acts = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = tgt_net.apply_softmax(next_distr_v)
    next_distr = next_distr.data.cpu().numpy()

    next_best_distr = next_distr[range(batch_size), next_acts]
    dones=np.asarray(dones)
    #dones = torch.Tensor.numpy(dones)
    dones = dones.astype(np.bool_)
    rewards=np.asarray(rewards)

    proj_distr = distr_projection(
        next_best_distr, rewards, dones, gamma)

    distr_v = net(states_v)
    sa_vals = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(sa_vals, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    loss_v = -state_log_sm_v * proj_distr_v
    return loss_v.sum(dim=1).mean()

def plot_stats(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title(f'Total frames {frame_idx}. Avg reward over last 10 episodes: {np.mean(rewards[-10:])}')
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig(MODEL_STATS_PATH+f"{frame_idx}-try.png")
    plt.show()
    plt.close()
    #stats_plot_inx+=1

def test(env,model,episodes,device):#(env, model, episodes, render=False, device=device, context=""):
    #model.eval()
    episode_rewards = []
    for episode in tqdm(range(episodes)):
        state = env.reset()
        episode_reward = 0.0
        while True:
            with torch.no_grad():
                action = model.act(state, 0, device)
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

            
    """
    #plot_stats
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title(f'Total episodes {episodes}. Avg reward episodes: {np.mean(episode_rewards[:])}')
    plt.plot(episode_rewards)
    #plt.subplot(132)
    #plt.title('loss')
    #plt.plot(losses)
    plt.savefig(f"./stats/test.png")
    plt.show()
    plt.close()
    """
    env.close()
    return episode_reward