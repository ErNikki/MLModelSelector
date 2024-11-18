import os
from SUN397 import Sun397
import torchvision
from utils import *
from pathlib import Path
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def collect_data(input_shape, train_dir, val_dir, test_dir, seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    train_dts = Sun397(
        img_dir=train_dir,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (input_shape[1], input_shape[2]), antialias=True
                ),
                torchvision.transforms.AutoAugment(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    val_dts = Sun397(
        img_dir=val_dir,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (input_shape[1], input_shape[2]), antialias=True
                ),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    test_dts = Sun397(
        img_dir=test_dir,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(
                    (input_shape[1], input_shape[2]), antialias=True
                ),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    print(
        f"Classes: {train_dts.num_classes}, \nTraining samples: {len(train_dts)}, \nVal sample: {len(val_dts)}, \nTest sample: {len(test_dts)}"
    )

    # create dataloaders
    train_data_loader = torch.utils.data.DataLoader(
        train_dts, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dts, batch_size=4, shuffle=True, num_workers=4, pin_memory=True
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dts, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_data_loader, val_data_loader, test_data_loader


def fit(
    epochs,
    lr,
    model,
    train_loader,
    val_loader,
    best_val_loss,
    best_val_acc,
    save_model_root,
):
    optimizer = torch.optim.Adam(model.parameters(), lr, amsgrad=False)
    # model = torch.compile(model) # torch 2.0
    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")
        # Training Phase
        model.train()
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        model.eval()
        result = evaluate(model, val_loader)
        val_loss, val_acc = model.epoch_end(epoch, result)
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), str(Path(save_model_root, "best_valLoss_model.pth")))
            best_val_loss = val_loss
            print(f"Best val_loss at epoch: {epoch}")
        if val_acc > best_val_acc:
            torch.save(model.state_dict(), str(Path(save_model_root, "best_valAcc_model.pth")))
            best_val_acc = val_acc
            print(f"Best val_acc at epoch: {epoch}")
    return best_val_loss, best_val_acc


def train_model(
    MODEL_NAME,
    input_shape,
    train_bool,
    training_params,
    train_data_loader,
    val_data_loader,
    test_data_loader,
):
    # Path
    env_path = Path("models/")
    save_path = Path(env_path, MODEL_NAME)
    save_path.mkdir(parents=True, exist_ok=True)
    # Hardware
    device = hardware_check()
    # Setup-train
    torch.cuda.empty_cache()
    best_val_loss, best_val_acc = float("inf"), 0
    # Build model
    if MODEL_NAME == "Efficientnet_b0":
        model = to_device(
            build_model(
                model=torchvision.models.efficientnet_b0(
                    weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "Efficientnet_b1":
        model = to_device(
            build_model(
                model=torchvision.models.efficientnet_b1(
                    weights=torchvision.models.EfficientNet_B1_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "Efficientnet_b3":
        model = to_device(
            build_model(
                model=torchvision.models.efficientnet_b3(
                    weights=torchvision.models.EfficientNet_B3_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "GoogLeNet":
        model = to_device(
            build_model(
                model=torchvision.models.googlenet(
                    weights=torchvision.models.GoogLeNet_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "Mnasnet0_5":
        model = to_device(
            build_model(
                model=torchvision.models.mnasnet0_5(
                    weights=torchvision.models.MNASNet0_5_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "Mnasnet0_75":
        model = to_device(
            build_model(
                model=torchvision.models.mnasnet0_75(
                    weights=torchvision.models.MNASNet0_75_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "Mnasnet0_10":
        model = to_device(
            build_model(
                model=torchvision.models.mnasnet1_0(
                    weights=torchvision.models.MNASNet1_0_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "MobNetV3S":
        model = to_device(
            build_model(
                model=torchvision.models.mobilenet_v3_small(
                    weights=torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "MobNetV3L":
        model = to_device(
            build_model(
                model=torchvision.models.mobilenet_v3_large(
                    weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "RegNet_x_400mf":
        model = to_device(
            build_model(
                model=torchvision.models.regnet_x_400mf(
                    weights=torchvision.models.RegNet_X_400MF_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "RegNet_x_800mf":
        model = to_device(
            build_model(
                model=torchvision.models.regnet_x_800mf(
                    weights=torchvision.models.RegNet_X_800MF_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "RegNet_y_400mf":
        model = to_device(
            build_model(
                model=torchvision.models.regnet_y_400mf(
                    weights=torchvision.models.RegNet_Y_400MF_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "RegNet_y_800mf":
        model = to_device(
            build_model(
                model=torchvision.models.regnet_y_800mf(
                    weights=torchvision.models.RegNet_Y_800MF_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "ResNet18":
        model = to_device(
            build_model(
                model=torchvision.models.resnet18(
                    weights=torchvision.models.ResNet18_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "ShufflNetV2_x05":
        model = to_device(
            build_model(
                model=torchvision.models.shufflenet_v2_x0_5(
                    weights=torchvision.models.ShuffleNet_V2_X0_5_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "ShufflNetV2_x10":
        model = to_device(
            build_model(
                model=torchvision.models.shufflenet_v2_x1_0(
                    weights=torchvision.models.ShuffleNet_V2_X1_0_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "ShufflNetV2_x15":
        model = to_device(
            build_model(
                model=torchvision.models.shufflenet_v2_x1_5(
                    weights=torchvision.models.ShuffleNet_V2_X1_5_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "ShufflNetV2_x20":
        model = to_device(
            build_model(
                model=torchvision.models.shufflenet_v2_x2_0(
                    weights=torchvision.models.ShuffleNet_V2_X2_0_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "Squeezenet1_0":
        model = to_device(
            build_model(
                model=torchvision.models.squeezenet1_0(
                    weights=torchvision.models.SqueezeNet1_0_Weights.DEFAULT
                )
            ),
            device,
        )
    elif MODEL_NAME == "Squeezenet1_1":
        model = to_device(
            build_model(
                model=torchvision.models.squeezenet1_1(
                    weights=torchvision.models.SqueezeNet1_1_Weights.DEFAULT
                )
            ),
            device,
        )
    else:
        raise NotImplementedError(f"{MODEL_NAME} has not been implemented yet!")
    
    # Compile model
    # Print model
    """print_model(
        model=model, device=device, save_model_root=save_path, input_shape=input_shape
    )"""
    # Moving data loader to appropriate device
    train_loader = DeviceDataLoader(train_data_loader, device)
    val_loader = DeviceDataLoader(val_data_loader, device)
    test_loader = DeviceDataLoader(test_data_loader, device)
    # Training
    if train_bool:
        for epochs, lr in training_params:
            print(f"Training for {epochs} epochs, with a learning rate of {lr}...")
            best_val_loss, best_val_acc = fit(
                epochs,
                lr,
                model,
                train_loader,
                val_loader,
                best_val_loss,
                best_val_acc,
                save_model_root=save_path,
            )
        print(
            "\n#----------------------#\n#  Process completed  #\n#----------------------#\n\n"
        )
    # Evaluate model
    else:
        """print(
            "\n#----------------------#\n#   Final Evaluation   #\n#----------------------#\n\n"
        )"""
        model.eval()
        print(str(Path(save_path, "best_valAcc_model.pth")))
        model_dict = torch.load(
            str(Path(save_path, "best_valAcc_model.pth")), map_location=torch.device(device)
        )
        model.load_state_dict(model_dict)
        final_result = evaluate(model, test_loader)
        _, _ = model.epoch_end(-1, final_result)


if __name__ == "__main__":
    # Dataset
    TRAIN_DATA = Path("./../dataset/SUN397/train_models")
    VAL_DATA = Path("./../dataset/SUN397/val_models")
    TEST_DATA = Path("./../dataset/SUN397/test")

    print(TRAIN_DATA)
    print(VAL_DATA)
    print(TEST_DATA)

    # Model
    #MODEL_NAME = "Efficientnet_b0"
    #MODEL_NAME = "ShufflNetV2_x05"
    #MODEL_NAME = "Squeezenet1_0"

    #MobNetV3S 0.6773
    #ResNet18  0.6684
    #ShufflNetV2_x05 0.6467
    
    #"ResNet18" 0.7013
    #"ShufflNetV2_x05" 0.6658
    #"ShufflNetV2_x10" 0.6959
    #"GoogLeNet" 0.7301
    #"MobNetV3S" 0.6946
    
    #models_name=["ResNet18","ShufflNetV2_x05","GoogLeNet","MobNetV3S"]
    MODELS_NAME=["Efficientnet_b3","GoogLeNet","ResNet18","MobNetV3S"]
    #MODELS_NAME=["Efficientnet_b3"]
    # Seed
    seed = 1000

    # Globals
    train_bool = False
    input_shape = (3, 362, 512)
    train_val_split = 0.1
    training_params = [(30, 1e-4), (15, 1e-5), (10, 1e-6), (5, 1e-7)]

    # Collect data
    train_data_loader, val_data_loader, test_data_loader = collect_data(
        input_shape=input_shape,
        train_dir=TRAIN_DATA,
        val_dir=VAL_DATA,
        test_dir=TEST_DATA,
        seed=seed,
    )
    
    for MODEL_NAME in MODELS_NAME:
        # Train model
        train_model(
            MODEL_NAME,
            input_shape,
            train_bool,
            training_params,
            train_data_loader,
            val_data_loader,
            test_data_loader,
        )
