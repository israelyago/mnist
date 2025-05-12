import os
import safetensors
import safetensors.torch
import torch
from torch.nn import functional as F
from torch import nn
from pathlib import Path
import logs

logger = logs.get_logger("model")


class UDLBookChapCNN(torch.nn.Module):

    def __init__(self):
        super(UDLBookChapCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        # Dropout for convolutions
        self.drop = nn.Dropout2d()
        # Fully connected layer
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        # 1
        x = self.conv1(x)
        # 2
        x = F.max_pool2d(x, 2)
        # 3
        x = F.relu(x)
        # 4
        x = self.conv2(x)
        # 5
        x = self.drop(x)
        # 6
        x = F.max_pool2d(x, 2)
        # 7
        x = F.relu(x)
        # 8
        x = x.flatten(1)
        # 9
        x = self.fc1(x)
        # 10
        x = F.relu(x)
        # 11
        x = self.fc2(x)
        # 12
        x = self.softmax(x)
        return x


class LeNet5(torch.nn.Module):
    """LeNet5 Inspired architecture."""

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(
            120, 10
        )  # Skipping Gaussian connections from original paper

        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        # 0
        padding_size = 2
        x = F.pad(
            x,
            (padding_size, padding_size, padding_size, padding_size),
            mode="constant",
            value=0,
        )
        # 1
        x = self.conv1(x)
        # 2
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # 3
        x = F.relu(x)
        # 4
        x = self.conv2(x)
        # 5
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # 6
        x = F.relu(x)
        # 7
        x = x.flatten(1)
        # 8
        x = self.fc1(x)
        # 9
        x = F.relu(x)
        # 10
        x = self.fc2(x)
        # 11
        x = self.softmax(x)
        return x


def load_model(file_name, arch, device="cpu"):

    model = UDLBookChapCNN()
    if arch == "lenet5":
        model = LeNet5()
    model = model.to(device)

    if file_name is not None:
        file = Path(file_name)
        if file.is_file():
            logger.info(f"✅ Model Checkpoint found in '{file}', loading tensors.")
            safetensors.torch.load_model(model, file_name, device=device)
    return model


def save_model(model, file_path_with_name: Path):
    os.makedirs(file_path_with_name.parent, exist_ok=True)
    safetensors.torch.save_model(model, file_path_with_name)
