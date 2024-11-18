from Ensemble import hard_voting, soft_voting
import torch
import torchvision
from utils import hardware_check, to_device, build_model, load_model
import os
from SUN397 import Sun397
from pathlib import Path
from EnsembleV2 import hard_voting_test_loader,soft_voting_test_loader
if __name__ == "__main__":
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

    test_data=Sun397(PATH_TEST_FOLDER, TRANSFORM_IMG)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False,  num_workers =4)
    
    val_data=Sun397(PATH_VAL_FOLDER, TRANSFORM_IMG)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False,  num_workers =4)

    #anche questa può essere una variabile in init
    env_path =Path("models/")
    #better first
    MODELS_NAME=["GoogLeNet","ResNet18","MobNetV3S"]
    #middle first 
    #MODELS_NAME=["ResNet18","GoogLeNet","MobNetV3S"] 
    #Worst first
    #MODELS_NAME=["MobNetV3S","GoogLeNet","ResNet18"]

    #OLD
    #MODELS_NAME=["MobNetV3S","ResNet18","ShufflNetV2_x05"]
    env_path =Path("models/")

    models_loaded=[]
    models_name_to_dict=[]
    index=0
    for model_name in MODELS_NAME:
        pth_path_model = Path(env_path, model_name)
        pth_path_model = os.path.join(pth_path_model, "best_valLoss_model.pth")
        model= load_model(model_name, pth_path_model, device)
        models_loaded.append(model)
        models_name_to_dict.append(model_name)

    """
    print("starting test data:")
    print("###################")
    hard_voting_test_loader(models_loaded, test_data_loader, device)
    soft_voting_test_loader(models_loaded, test_data_loader, device)
    """
    
    print("starting val data:")
    print("###################")
    hard_voting_test_loader(models_loaded, val_data_loader, device)
    soft_voting_test_loader(models_loaded, val_data_loader, device)
