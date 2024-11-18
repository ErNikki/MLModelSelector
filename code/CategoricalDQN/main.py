import torch
import torchvision
import torch.optim as optim

#from utils2 import *

from DistributionalDQN import DistributionalDQN
from ParamDDQN import *
from Loss import train_double, test, train_doublev2

from pathlib import Path
import os

import sys
sys.path.append('../')

from Environment import _NetSelectorEnv
from SUN397 import Sun397
from replay_memory import ReplayBuffer
from utils import hardware_check


if __name__ == "__main__":

    train_mode=False
    trainv2=True
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

    if train_mode and not trainv2:
        
        print(PATH_TRAIN_FOLDER)
        train_data=Sun397(PATH_TRAIN_FOLDER,TRANSFORM_IMG)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=1024, shuffle=True,  num_workers =2, prefetch_factor=1)
        
        MODELS_NAME=["MobNetV3S","ResNet18","ShufflNetV2_x05"]
        
        #print(os.getcwd()+"/../models/")
        env=_NetSelectorEnv(MODELS_NAME, train_data_loader, device,1024, train_mode, env_path=os.getcwd()+"/../models/")
        

        #observations=input action_space=output
        current_model = DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)
        target_model= DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)
        target_model.load_state_dict(current_model.state_dict())
        
        optimizer = optim.Adam(current_model.parameters(), lr=0.00001)
        
        replay_buffer = ReplayBuffer(MEMORY_SIZE)
        train_double(env, current_model, target_model, optimizer, replay_buffer, device)
        
    elif train_mode and trainv2:
        train_data=Sun397(PATH_TRAIN_FOLDER,TRANSFORM_IMG)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=2048, shuffle=True,  num_workers = 2, prefetch_factor=1 )
        
        val_data=Sun397(PATH_VAL_FOLDER,TRANSFORM_IMG)
        val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=3606, shuffle=True,  num_workers =1, prefetch_factor=1)
        
        MODELS_NAME=["MobNetV3S","ResNet18","ShufflNetV2_x05"]
        print("#### Building envs ####")
        
        train_env=_NetSelectorEnv(MODELS_NAME, val_data_loader, device, 3606, train_mode= False, env_path=os.getcwd()+"/../models/")
        env=_NetSelectorEnv(MODELS_NAME, train_data_loader, device, 2048, train_mode, env_path=os.getcwd()+"/../models/")
        
        #train_env=_NetSelectorEnv(MODELS_NAME, val_data_loader, device, 3606, train_mode= False)
        #env=_NetSelectorEnv(train_data_loader,device,512,train_mode)
        
        print("#### Building models ####")
        #observations=input action_space=output
        current_model = DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)
        target_model= DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)
        target_model.load_state_dict(current_model.state_dict())
        
        
        
        optimizer = optim.Adam(current_model.parameters(), lr=0.00001)
        replay_buffer = ReplayBuffer(MEMORY_SIZE)
        print("#### starting train envs ####")
        train_doublev2(env, train_env, current_model, target_model, optimizer, replay_buffer, device)
        
    else:
        
        test_data=Sun397(PATH_TEST_FOLDER,TRANSFORM_IMG)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=3920,  shuffle=True,  num_workers =1, prefetch_factor=1)
        
        MODELS_NAME=["MobNetV3S","ResNet18","ShufflNetV2_x05"]
        
        env=_NetSelectorEnv(MODELS_NAME, test_data_loader, device,3920, train_mode,env_path=os.getcwd()+"/../models/")

    
        #observations=input action_space=output
        model = DistributionalDQN(env.observation_space.shape, env.action_space.n)
        model.to(device)
        model.eval()
        
        start=66
        for i in range(0,1):
            
            i=i*100
            #env=_NetSelectorEnv(MODELS_NAME, test_data_loader, device,3920, train_mode)
            #MODEL_NAME="netSelector_curr_episode_"+str(i+start)+".pth"
            MODEL_NAME="netSelector_curr_episode_"+str(66)+".pth"
            env_path=os.getcwd()+"/../rf-models/x2trainDistributional/"
            pth_path_model = Path(env_path, MODEL_NAME)
            
            #model = torchvision.models.regnet_y_800mf(weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT)
            
            model.load_state_dict(torch.load(pth_path_model))
            
            model.eval()
            print(" ")
            print("###################################")
            print(MODEL_NAME)
            #with torch.no_grad:
            test(env, model, 1, device)
            
            #env=_NetSelectorEnv(MODELS_NAME, test_data_loader, device,3920, train_mode)
            #MODEL_NAME="netSelector_target_episode_"+str(i+start)+".pth"
            MODEL_NAME="netSelector_target_episode_"+str(39)+".pth"
            env_path=os.getcwd()+"/../rf-models/x2trainDistributional/"
            pth_path_model = Path(env_path, MODEL_NAME)
            
            #model = torchvision.models.regnet_y_800mf(weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT)
            #model = build_modelV2(model,len(MODELS_NAME))
            model.load_state_dict(torch.load(pth_path_model))
            #model = to_device(model, device)
            model.eval()
            print(" ")
            print("###################################")
            print(MODEL_NAME)
            #with torch.no_grad:
            test(env, model, 1, device)
