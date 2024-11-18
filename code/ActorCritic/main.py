import gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

import torchvision

from lib import common

import os
import sys
sys.path.append('../')

from Environment import _NetSelectorEnv
from SUN397 import Sun397
from replay_memory import ReplayBuffer
from utils import hardware_check
from utils2 import test

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1


class AtariA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)
    
    def unpack_batch(batch, net, device='cpu'):
        """
        Convert batch into training tensors
        :param batch:
        :param net:
        :return: states variable, actions tensor, reference values variable
        """
        states = []
        actions = []
        rewards = []
        not_done_idx = []
        last_states = []
        for idx, exp in enumerate(batch):
            states.append(np.array(exp.state, copy=False))
            actions.append(int(exp.action))
            rewards.append(exp.reward)
            if exp.last_state is not None:
                not_done_idx.append(idx)
                last_states.append(np.array(exp.last_state, copy=False))

        states_v = torch.FloatTensor(
            np.array(states, copy=False)).to(device)
        actions_t = torch.LongTensor(actions).to(device)

        # handle rewards
        rewards_np = np.array(rewards, dtype=np.float32)
        if not_done_idx:
            last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
            last_vals_v = net(last_states_v)[1]
            last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
            last_vals_np *= GAMMA ** REWARD_STEPS
            rewards_np[not_done_idx] += last_vals_np

        ref_vals_v = torch.FloatTensor(rewards_np).to(device)

        return states_v, actions_t, ref_vals_v

    if __name__ == "__main__":
        
        train_mode=False
        double_train=True
        device = hardware_check()
        PATH_DATASET_FOLDER = os.getcwd() + "/../../dataset/SUN397/"
        PATH_TRAIN_FOLDER=PATH_DATASET_FOLDER+  "train_selector"
        PATH_TEST_FOLDER=PATH_DATASET_FOLDER+"/test"
        PATH_VAL_FOLDER=PATH_DATASET_FOLDER+"/val_selector"
        
        
        TRANSFORM_IMG = torchvision.transforms.Compose([
            #torchvision.transforms.Resize(256),
            #torchvision.transforms.CenterCrop(256),
            torchvision.transforms.Resize((362, 512), antialias=True),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225] )
            ])
        
        print(PATH_TRAIN_FOLDER)
        train_data=Sun397(PATH_TRAIN_FOLDER,TRANSFORM_IMG)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1024, shuffle=True,  num_workers =2, prefetch_factor=1)
        
        MODELS_NAME=["MobNetV3S","ResNet18","ShufflNetV2_x05"]
        
        #print(os.getcwd()+"/../models/")
        env=_NetSelectorEnv(MODELS_NAME, train_data_loader, device,1024, train_mode, env_path=os.getcwd()+"/../models/")
        
        #net = AtariA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)