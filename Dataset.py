import cv2
import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, file_lists, label_lists, transforms=None):
        self.file_lists = file_lists.copy()
        self.label_lists = label_lists.copy()
        self.transforms = transforms

    def __getitem__(self, idx):
        img = cv2.imread(self.file_lists[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        img = img.transpose(2, 0, 1)

        label = self.label_lists[idx]

        img = torch.tensor(img, dtpye=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return img, label

    def __len__(self):
        assert len(self.file_lists) == len(self.label_lists)
        return len(self.file_lists)


class TestDataset(Dataset):
    def __init__(self, file_lists, transforms=None):
        self.file_lists = file_lists.copy()
        self.transforms = transforms

    def __getitem__(self, idx):
        img = cv2.imread(self.file_lists[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        img = img.transpose(2, 0, 1)

        img = torch.tensor(img, dtpye=torch.float)

        return img

    def __len__(self):
        return len(self.file_lists)
