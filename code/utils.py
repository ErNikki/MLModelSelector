from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
#from torchsummaryX import summary
from torchsummary import summary
import torchvision
import random
import numpy as np

def time_str():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


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


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


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


def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');


def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


def print_model(model, device, save_model_root, input_shape):
    info = summary(model, torch.zeros((1, input_shape[0], input_shape[1], input_shape[2])).to(device))
    #info.to_csv(save_model_root + 'model_summary.csv')

@torch.no_grad()
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


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
        
        model_dict = torch.load(
            str(Path(pth_path)), map_location=torch.device(device)
        )
        model.load_state_dict(model_dict)
        

        # set to eval mode
        model.eval()

        # freeze layers
        for parameter in model.parameters():
            parameter.requires_grad = False

        return model