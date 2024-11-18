from collections import deque
import random
import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        #state = state.unsqueeze(0)
        #next_state = next_state.unsqueeze(0)
        state = np.expand_dims(state.cpu(), 0)
        next_state = np.expand_dims(next_state.cpu(), 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        #in realtà random non è utile nel nostro caso siccome le immagini sono tutte indipendenti
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        #return torch.cat(state,0), action, reward, torch.cat(next_state,0), done
        return np.concatenate(state), action, reward, np.concatenate(next_state), done
        
    def __len__(self):
        return len(self.buffer)