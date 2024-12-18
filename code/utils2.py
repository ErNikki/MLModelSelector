import os
import numpy as np
import torch
import gym
from Param import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#stats_plot_inx=0

def plot_stats(frame_idx, rewards, losses):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title(f'Total frames {frame_idx}. Avg reward over last 10 episodes: {np.mean(rewards[-10:])}')
    plt.plot(rewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(losses)
    plt.savefig(MODEL_STATS_PATH+f"{frame_idx}try.png")
    plt.show()
    plt.close()
    #stats_plot_inx+=1


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

def train(env, model, optimizer, replay_buffer, device=device):
    steps_done = 0
    episode_rewards = []
    losses = []
    model.train()
    for episode in tqdm(range(EPISODES)):
        state = env.reset()
        episode_reward = 0.0
        while True:
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(- steps_done / EPS_DECAY)
            action = model.act(state, epsilon, device)
            steps_done += 1

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(replay_buffer) > INITIAL_MEMORY:
                loss = compute_loss(model, replay_buffer, BATCH_SIZE, GAMMA, device)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            #ogni 10 step fa vedere le stats
            if steps_done % 1024 == 0:
                plot_stats(steps_done, episode_rewards, losses)

            if done:
                episode_rewards.append(episode_reward)
                break
            
        if (episode+1) % 100 == 0:
            path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_episode_{episode+1}.pth")
            print(f"Saving weights at Episode {episode+1} ...")
            torch.save(model.state_dict(), path)
    env.close()


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
    right_guesses=env.close()
    return right_guesses
    


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


def train_double(env, current_model,target_model, optimizer, replay_buffer, device=device):
    steps_done = 0
    episode_rewards = []
    losses = []
    current_model.train()
    for episode in range(EPISODES):
        state = env.reset()
        episode_reward = 0.0
        while True:
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(- steps_done / EPS_DECAY)
            action = current_model.act(state, epsilon, device)
            steps_done += 1

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if len(replay_buffer) > INITIAL_MEMORY:
                loss = compute_loss_double(current_model, target_model, replay_buffer, BATCH_SIZE, GAMMA, device)

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
            epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(- steps_done / EPS_DECAY)
            action = current_model.act(state, epsilon, device)
            steps_done += 1

            next_state, reward, done, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            #episode_reward += reward

            if len(replay_buffer) > INITIAL_MEMORY :
                
                loss = compute_loss_double(current_model, target_model, replay_buffer, BATCH_SIZE, GAMMA, device)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if steps_done % UPDATE_TARGET == 0 and steps_done > WARMUP:
                target_model.load_state_dict(current_model.state_dict())

            if steps_done % PLOT_STATS == 0 and steps_done > WARMUP:
                plot_stats(steps_done, episode_rewards, losses)

            
            if done:
                episode_rewards.append(episode_reward)
                break
            
            
        if (episode+1) % 1 == 0 and steps_done > WARMUP:
            print("TESTING EPISODE CURRENT")
            result=test(test_env, current_model, 1, device)
            if result>best_reward_curr_model:
                best_reward_curr_model=result
                curr_path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_curr_episode_{episode+1}.pth")
                print(f"Saving curr weights at Episode {episode+1} ...")
                torch.save(current_model.state_dict(), curr_path)
                
            episode_rewards.append(result)
            print("TESTING EPISODE TARGET")

            result=test(test_env, target_model, 1, device)
            if result>best_reward_target_model:
                best_reward_target_model=result
                target_path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_target_episode_{episode+1}.pth")
                print(f"Saving target weights at Episode {episode+1} ...")
                torch.save(target_model.state_dict(), target_path)
            
    env.close()

def train_doublex2(env, current_model,target_model, optimizer, replay_buffer, device=device):
    print("starting x2 train")
    steps_done = 0
    episode_rewards = []
    losses = []
    current_model.train()
    for episode in tqdm(range(EPISODES)):
        state = env.reset()
        #episode_reward = 0.0
        while True:
            #epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(- steps_done / EPS_DECAY)
            #action = current_model.act(state, epsilon, device)
            steps_done += 1

            next_state, rewards, done, _ = env.step(None)
            
            i=0
            for r in rewards:
                replay_buffer.push(state, i, r, next_state, done)
                i+=1

            state = next_state
            #episode_reward += reward

            if len(replay_buffer) > INITIAL_MEMORY:
                loss = compute_loss_double(current_model, target_model, replay_buffer, BATCH_SIZE, GAMMA, device)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if steps_done % 128 == 0:
                target_model.load_state_dict(current_model.state_dict())

            #if steps_done % 1024 == 0:
            #    plot_stats(steps_done, episode_rewards, losses)
            
            if done:
                #episode_rewards.append(episode_reward)
                break
            
        if (episode+1) % 100 == 0:
            curr_path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_curr_episode_{episode+1}.pth")
            target_path = os.path.join(MODEL_SAVE_PATH, f"{env.spec_id}_target_episode_{episode+1}.pth")
            print(f"Saving weights at Episode {episode+1} ...")
            torch.save(current_model.state_dict(), curr_path)
            torch.save(target_model.state_dict(), target_path)
    env.close()

def train_doublev2x2(env, test_env, current_model, target_model, optimizer, replay_buffer, device=device):
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
            #action = current_model.act(state, 1, device)
            steps_done += 1

            next_state, rewards, done, _ = env.step(None)
            
            i=0
            for r in rewards:
                replay_buffer.push(state, i, r, next_state, done)
                i+=1

            state = next_state
            #replay_buffer.push(state, action, reward, next_state, done)

            #state = next_state
            #episode_reward += reward

            if len(replay_buffer) > INITIAL_MEMORY:
                
                loss = compute_loss_double(current_model, target_model, replay_buffer, BATCH_SIZE, GAMMA, device)

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if steps_done % 512 == 0:
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