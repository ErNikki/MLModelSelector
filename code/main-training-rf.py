import torch
import torchvision
from Environment import _NetSelectorEnv
from utils2 import *
from utils import hardware_check, to_device, build_modelV2
from RFmodel import DQN
from replay_memory import ReplayBuffer
from Param import *
import torch.optim as optim
from SUN397 import Sun397
from pathlib import Path
from CategoricalDQN.DistributionalDQN import DistributionalDQN

if __name__ == "__main__":

    train_mode=False
    double_train=True
    x2_train=True
    device = hardware_check()
    PATH_DATASET_FOLDER = os.getcwd() + "/../dataset/SUN397/"
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
    
    if train_mode and not double_train and not x2_train:
        print(PATH_TRAIN_FOLDER)
        train_data=Sun397(PATH_TRAIN_FOLDER,TRANSFORM_IMG)
        #test_data=Sun397(PATH_TEST_FOLDER,TRANSFORM_IMG)
        #test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=512,  num_workers =4, prefetch_factor=1)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True,  num_workers =4, prefetch_factor=1)
        
        MODELS_NAME=["MobNetV3S","ResNet18","ShufflNetV2_x05"]
        
        env=_NetSelectorEnv(MODELS_NAME, train_data_loader, device,512, train_mode)
        #env=_NetSelectorEnv(train_data_loader,device,512,train_mode)
        

        #observations=input action_space=output
        #model = DQN(env.observation_space.shape, env.action_space.n).to(device)
        
        #AGENT_MODEL="Efficientnet_b0"
        
        #model=torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        #model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
        model = torchvision.models.regnet_y_800mf(weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT)
        
        model=build_modelV2(model,len(MODELS_NAME))
        model = to_device(model, device)
        optimizer = optim.Adam(model.parameters(), lr=0.00001)
        
        replay_buffer = ReplayBuffer(MEMORY_SIZE)
        train(env, model, optimizer, replay_buffer, device)
        
    elif train_mode and double_train and not x2_train:
        print(PATH_TRAIN_FOLDER)
        train_data=Sun397(PATH_TRAIN_FOLDER,TRANSFORM_IMG)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True,  num_workers =4, prefetch_factor=1 )
        
        MODELS_NAME=["MobNetV3S","ResNet18","ShufflNetV2_x05"]
        env=_NetSelectorEnv(MODELS_NAME, train_data_loader, device, 512, train_mode)
        #env=_NetSelectorEnv(train_data_loader,device,512,train_mode)
        

        #observations=input action_space=output
        #current_model = DQN(env.observation_space.shape, env.action_space.n).to(device)
        #target_model = DQN(env.observation_space.shape, env.action_space.n).to(device)
        
        current_model = torchvision.models.regnet_y_800mf(weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT)
        current_model = build_modelV2(current_model,len(MODELS_NAME))
        
        
        target_model = torchvision.models.regnet_y_800mf(weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT)
        target_model = build_modelV2(target_model,len(MODELS_NAME))
        target_model.load_state_dict(current_model.state_dict())
        
        target_model = to_device(target_model, device)
        current_model = to_device(current_model, device)
        
        optimizer = optim.Adam(current_model.parameters(), lr=0.00001)
        replay_buffer = ReplayBuffer(MEMORY_SIZE)
        train_double(env, current_model, target_model, optimizer, replay_buffer, device)
    
    elif train_mode and double_train and x2_train:
        print(PATH_TRAIN_FOLDER)
        train_data=Sun397(PATH_TRAIN_FOLDER,TRANSFORM_IMG)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=512, shuffle=True,  num_workers =2, prefetch_factor=1 )
        
        MODELS_NAME=["MobNetV3S","ResNet18","ShufflNetV2_x05"]
        env=_NetSelectorEnv(MODELS_NAME, train_data_loader, device, 512, train_mode, x2_train)
        #env=_NetSelectorEnv(train_data_loader,device,512,train_mode)
        

        #observations=input action_space=output
        #current_model = DQN(env.observation_space.shape, env.action_space.n).to(device)
        #target_model = DQN(env.observation_space.shape, env.action_space.n).to(device)
        
        
        current_model = torchvision.models.regnet_y_400mf(weights=torchvision.models.RegNet_Y_400MF_Weights.DEFAULT)
        current_model = build_modelV2(current_model,len(MODELS_NAME))
        
        target_model = torchvision.models.regnet_y_400mf(weights=torchvision.models.RegNet_Y_400MF_Weights.DEFAULT)
        target_model = build_modelV2(target_model,len(MODELS_NAME))
        target_model.load_state_dict(current_model.state_dict())
        
        target_model = to_device(target_model, device)
        current_model = to_device(current_model, device)
        
        
        optimizer = optim.Adam(current_model.parameters(), lr=0.00001)
        replay_buffer = ReplayBuffer(MEMORY_SIZE)
        train_doublex2(env, current_model, target_model, optimizer, replay_buffer, device)
    
    else:
        
        test_data=Sun397(PATH_TEST_FOLDER,TRANSFORM_IMG)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=3920, shuffle=True,  num_workers =1, prefetch_factor=1)
        
        train_data=Sun397(PATH_TRAIN_FOLDER,TRANSFORM_IMG)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=3920, shuffle=True,  num_workers =1, prefetch_factor=1)
        
        val_data=Sun397(PATH_VAL_FOLDER,TRANSFORM_IMG)
        val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=3606, shuffle=True,  num_workers =1, prefetch_factor=1)
        
        MODELS_NAME=["MobNetV3S","ResNet18","ShufflNetV2_x05"]
        
        env=_NetSelectorEnv(MODELS_NAME, val_data_loader, device,3606, train_mode)

        
        #observations=input action_space=output
        model = DQN(env.observation_space.shape, env.action_space.n)        
        model.to(device)
        model.eval()
        
        
        #AGENT_MODEL="Efficientnet_b0"
        #model=torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        #model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
        #model = torchvision.models.regnet_y_800mf(weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT)
        
        #model = build_modelV2(model,len(MODELS_NAME))
        #model.load_state_dict(torch.load(pth_path_model))
        #model = to_device(model, device)
        #model.eval()
        
        """
        #model = torchvision.models.regnet_y_400mf(weights=torchvision.models.RegNet_Y_400MF_Weights.DEFAULT)
        model = torchvision.models.regnet_y_800mf()
        model = build_modelV2(model,len(MODELS_NAME))
        model = to_device(model, device)
        model.eval()
        """
        
        
    
        #model = build_modelV2(model,len(MODELS_NAME))
        #model = to_device(model, device)
        
        start=50
        for i in range(0,38):
            
            i=i*50
            #env=_NetSelectorEnv(MODELS_NAME, test_data_loader, device,3920, train_mode)
            MODEL_NAME="netSelector_curr_episode_"+str(i+start)+".pth"
            env_path = Path("./rf-models/x2train/")
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
            MODEL_NAME="netSelector_target_episode_"+str(i+start)+".pth"
            env_path = Path("./rf-models/x2train/")
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
