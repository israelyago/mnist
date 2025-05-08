from torch.utils.data import Dataset
import torch
from torch import nn
from PIL import Image
import io
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, dataframe, device):
        self.dataframe = dataframe
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        img = self._get_image_tensor_from_row(row)
        label = self._get_label_one_hotted_from_row(row)
        return img, label

    def _get_image_tensor_from_row(self, row):
        img_field = row["image"]
        img = Image.open(io.BytesIO(img_field["bytes"]))
        pixels = list(img.getdata())
        pixels = np.array(pixels)
        grayscale_tensor = torch.from_numpy(pixels).to(
            device=self.device, dtype=torch.float32
        )

        return grayscale_tensor

    def _get_label_one_hotted_from_row(self, row):
        label = torch.tensor(row["label"], device=self.device)
        return nn.functional.one_hot(label, num_classes=10).float()
