import torch
import torchvision
import torch.optim as optim
from pathlib import Path

from TestEnv import TestEnv
from TrainEnv import TrainEnv
from newTestEnv import newTestEnv
from newTrainEnv import newTrainEnv
from newTestEnvSimplified import newTestEnvSimplified
from newTrainEnvSimplified import newTrainEnvSimplified
from newTestEnvSimplifiedwithCumulativeRew import newTestEnvSimplifiedCR
from newTrainEnvSimplifiedwithCumulativeRew import newTrainEnvSimplifiedCR
from utils import hardware_check, to_device, build_modelV2, test, train_doublev2
#from RFmodel import DQN
from replay_memory import ReplayBuffer
from Param import *
from SUN397 import Sun397

if __name__ == "__main__":
    
    train_mode=False
    SELECTED_ENV=1
    device = hardware_check()
    
    TRANSFORM_IMG = torchvision.transforms.Compose([
        #torchvision.transforms.Resize(256),
        #torchvision.transforms.CenterCrop(256),
        torchvision.transforms.Resize((362, 512), antialias=True),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )
        ])
    train_data=Sun397(PATH_TRAIN_FOLDER,TRANSFORM_IMG)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=EPISODES_LENGHT, shuffle=True,  num_workers = 0)#, prefetch_factor=1 )

    val_data=Sun397(PATH_VAL_FOLDER,TRANSFORM_IMG)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=3606, shuffle=False)#,  num_workers =1, prefetch_factor=1)
    #"ResNet18" 0.7013
    #"ShufflNetV2_x05" 0.6658
        #"ShufflNetV2_x10" 0.6959
    #"GoogLeNet" 0.7301
    #"MobNetV3S" 0.6946
    #MODELS_NAME=["MobNetV3S","ResNet18"]#,"ShufflNetV2_x05"]
    #changed for simpleEnv
    #MODELS_NAME=["ResNet18","ShufflNetV2_x05","MobNetV3S"]
    #MODELS_NAME=["MobNetV3S","ResNet18","GoogLeNet"]
    MODELS_NAME=["GoogLeNet","ResNet18","MobNetV3S"]  
    current_model = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)
    target_model = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)
    
    if train_mode:
        print("#### Building envs ####")
        print()
        #print(PATH_TRAIN_FOLDER)
        match SELECTED_ENV:
            case 1:
                print("building main env:")
                train_env=TrainEnv(MODELS_NAME, train_data_loader, device, EPISODES_LENGHT)
                print("building test env:")
                test_env=TestEnv(MODELS_NAME, val_data_loader, device, 3606)
                
                print("#### Building models ####")
                
                current_model = build_modelV2(current_model, len(MODELS_NAME))
                
                target_model = build_modelV2(target_model, len(MODELS_NAME))
                #HANNO GIà LE STESSE WEIGHT ME NEL DUBBIO MEGLIO FARLO
                target_model.load_state_dict(current_model.state_dict())
                
                target_model = to_device(target_model, device)
                current_model = to_device(current_model, device)
                
            case 2:
                print("building main env:")
                train_env=newTrainEnv(MODELS_NAME, train_data_loader, device, EPISODES_LENGHT)
                print("building test env:")
                test_env=newTestEnv(MODELS_NAME, val_data_loader, device, 3606)
                
                print("#### Building models ####")
                current_model = build_modelV2(current_model, len(MODELS_NAME)+1)
                
                target_model = build_modelV2(target_model, len(MODELS_NAME)+1)
                #HANNO GIà LE STESSE WEIGHT ME NEL DUBBIO MEGLIO FARLO
                target_model.load_state_dict(current_model.state_dict())
                
                target_model = to_device(target_model, device)
                current_model = to_device(current_model, device)
                
            case 3:
                print("building test env:")
                train_env=newTrainEnvSimplified(MODELS_NAME, train_data_loader, device, EPISODES_LENGHT)
                print("building test env:")
                test_env=newTestEnvSimplified(MODELS_NAME, val_data_loader, device, 3606)
        
                print("#### Building models ####")
                current_model = build_modelV2(current_model, len(MODELS_NAME))
                
                target_model = build_modelV2(target_model, len(MODELS_NAME))
                #HANNO GIà LE STESSE WEIGHT ME NEL DUBBIO MEGLIO FARLO
                target_model.load_state_dict(current_model.state_dict())
                
                target_model = to_device(target_model, device)
                current_model = to_device(current_model, device)
                
            
    
        
        
        print("#### Building models ####")
        #observations=input action_space=output
        
        """
        current_model = DQN(env.observation_space.shape, env.action_space.n).to(device)
        target_model = DQN(env.observation_space.shape, env.action_space.n).to(device)
        target_model.load_state_dict(current_model.state_dict())
        """
        
        """
        current_model = torchvision.models.regnet_y_400mf(weights=torchvision.models.RegNet_Y_400MF_Weights.DEFAULT)
        current_model = build_modelV2(current_model, len(MODELS_NAME))
        
        target_model = torchvision.models.regnet_y_400mf(weights=torchvision.models.RegNet_Y_400MF_Weights.DEFAULT)
        target_model = build_modelV2(target_model, len(MODELS_NAME))
        target_model.load_state_dict(current_model.state_dict())
        
        target_model = to_device(target_model, device)
        current_model = to_device(current_model, device)
        """
        
        
        
        
        """
        current_model = torchvision.models.efficientnet_b5(weights=torchvision.models.EfficientNet_B5_Weights.DEFAULT)
        current_model = build_modelV2(current_model, len(MODELS_NAME))
        
        target_model = torchvision.models.efficientnet_b5(weights=torchvision.models.EfficientNet_B5_Weights.DEFAULT)
        target_model = build_modelV2(target_model, len(MODELS_NAME))
        target_model.load_state_dict(current_model.state_dict())
        
        target_model = to_device(target_model, device)
        current_model = to_device(current_model, device)
        """
        optimizer = optim.Adam(current_model.parameters(), lr=0.00025)
        replay_buffer = ReplayBuffer(MEMORY_SIZE)
        
        
        print("#### starting train envs ####")
        #train_doublev2(env, train_env, current_model, target_model, optimizer, replay_buffer, device)
        train_doublev2(train_env, test_env, current_model, target_model, optimizer, replay_buffer, device)
    
    #test
    else:
        """
        need to modify:
        current=[]
        target=[]
        CURRENT_MODEL_FILE=
        TARGET_MODEL_FILE=
        env_path = 
        """
        
        test_data=Sun397(PATH_TEST_FOLDER,TRANSFORM_IMG)
        test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=3920, shuffle=True,  num_workers =1, prefetch_factor=1)

        train_data=Sun397(PATH_TRAIN_FOLDER,TRANSFORM_IMG)
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=3920, shuffle=True,  num_workers =1, prefetch_factor=1)

        val_data=Sun397(PATH_VAL_FOLDER,TRANSFORM_IMG)
        val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=3606, shuffle=False,  num_workers =1, prefetch_factor=1)

        val_models_data=Sun397(PATH_VAL_MODELS_FOLDER,TRANSFORM_IMG)
        val_models_data_loader = torch.utils.data.DataLoader(val_models_data, batch_size=3588, shuffle=True,  num_workers =1, prefetch_factor=1)

        #MODELS_NAME=["MobNetV3S","ResNet18","ShufflNetV2_x05"]
        MODELS_NAME=["GoogLeNet","ResNet18","MobNetV3S"]
        
        #chosen_subset=test_data_loader
        #n_images=3920
        
        chosen_subset=val_data_loader
        n_images=3606
        
        match SELECTED_ENV:
            case 1:
                print("building test env:")
                env=TestEnv(MODELS_NAME, chosen_subset, device, n_images)
                
                print("#### Building models ####")
                model = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)
                model = build_modelV2(model, len(MODELS_NAME))
                model = to_device(model, device)
                model.eval()
                
                
            case 2:
                print("building test env:")
                env=newTestEnv(MODELS_NAME, chosen_subset, device, n_images)
                
                print("#### Building models ####")
                model = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)
                model = build_modelV2(model, len(MODELS_NAME)+1)
                model = to_device(model, device)
                model.eval()
                
            case 3:
                print("building test env:")
                env=newTestEnvSimplified(MODELS_NAME, chosen_subset, device, n_images)
                
                print("#### Building models ####")
                model = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)
                model = build_modelV2(model, len(MODELS_NAME))
                model = to_device(model, device)
                model.eval()
                



        #observations=input action_space=output
        """
        model = DQN(env.observation_space.shape, env.action_space.n)        
        model.to(device)
        model.eval()
        """



        #AGENT_MODEL="Efficientnet_b0"
        #model=torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.DEFAULT)
        #model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)
        #model = torchvision.models.regnet_y_800mf(weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT)

        #model = build_modelV2(model,len(MODELS_NAME))
        #model.load_state_dict(torch.load(pth_path_model))
        #model = to_device(model, device)
        #model.eval()
        """
        model = torchvision.models.efficientnet_b3(weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT)
        model = build_modelV2(model, len(MODELS_NAME))
        """
        #model = torchvision.models.regnet_y_400mf(weights=torchvision.models.RegNet_Y_400MF_Weights.DEFAULT)
        #model = torchvision.models.regnet_y_800mf()
        #model = build_modelV2(model,len(MODELS_NAME))
        """
        model = to_device(model, device)
        model.eval()
        """

        #model = build_modelV2(model,len(MODELS_NAME))
        #model = to_device(model, device)
        
        save_stats=False
        current = [2,3,40]
        target = []
        
        CURRENT_MODEL_FILE = "netSelector_curr_episode_"
        TARGET_MODEL_FILE = "netSelector_target_episode_"
        
        #path_stats_current="./rf-models/efficientnet_b3/TrainEnvV2/test_stats/current/"
        #path_stats_target="./rf-models/efficientnet_b3/TrainEnvV2/test_stats/target/"
        
        path_stats_current = MODEL_STATS_TEST_PATH+"current/"
        path_stats_target = MODEL_STATS_TEST_PATH+"target/"

        #start=50
        #for i in range(0,1):
        for i in current:
            
            #i=i*50
            #env=_NetSelectorEnv(MODELS_NAME, test_data_loader, device,3920, train_mode)
            #MODEL_NAME="netSelector_curr_episode_"+str(i+start)+".pth"
            MODEL_NAME=CURRENT_MODEL_FILE+str(i)+".pth"
            #env_path = Path("./rf-models/efficientnet_b3/TrainEnvV2/")
            #env_path = Path("./rf-models/efficientnet_b3/TrainEnvV2/")
            env_path = Path(MODEL_SAVE_PATH)
            pth_path_model = Path(env_path, MODEL_NAME)
            
            #model = torchvision.models.regnet_y_800mf(weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT)
            
            model.load_state_dict(torch.load(pth_path_model))
            
            model.eval()
            print(" ")
            print("###################################")
            print(MODEL_NAME)
            #with torch.no_grad:
            
            test(env, model, 1, device, (save_stats, path_stats_current+f"{i}/"))
            
        for i in target:   
            
            #env=_NetSelectorEnv(MODELS_NAME, test_data_loader, device,3920, train_mode)
            #MODEL_NAME="netSelector_target_episode_"+str(i+start)+".pth"
            MODEL_NAME=TARGET_MODEL_FILE+str(i)+".pth"
            
            #env_path = Path("./rf-models/efficientnet_b3/TrainEnvV2/")
            #env_path = Path("./rf-models/efficientnet_b3/TrainEnvV2/")
            env_path = Path(MODEL_SAVE_PATH)
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
            test(env, model, 1, device,(save_stats, path_stats_target+f"{i}/"))
