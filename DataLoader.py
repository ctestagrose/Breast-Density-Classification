import torch
from monai.data import Dataset
import cv2


class dataloader(Dataset):
    def __init__(self, dict, transforms):
        self.dict = dict
        self.transforms = transforms

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, index):
        image = cv2.imread(self.dict[index]['image'])
        image = self.transforms(image)
        label = self.dict[index]['label']
        label = torch.FloatTensor(label)
        return image, label

class val_dataloader(Dataset):
    def __init__(self, dict, transforms):
        self.dict = dict
        self.transforms = transforms

    def __len__(self):
        return len(self.dict)

    def __getitem__(self, index):
        image = cv2.imread(self.dict[index]['image'])
        image = self.transforms(image)
        label = self.dict[index]['label']
        label = torch.FloatTensor(label)
        return image, label