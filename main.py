import albumentations as A
import torchvision
from sklearn.model_selection import train_test_split, StratifiedKFold
import img_encoder
import torch
import numpy as np
import os
import random
from Gan_Train import Train
from torch.utils.data import DataLoader
from Dataset import TrainDataset

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

dir_name = 'Test_data_set'
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

num_workers = 2
batch_size = 48

train_transforms = A.Compose([
    A.Rotate(),
    A.HorizontalFlip(),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    A.Normalize()
])
val_transforms = A.Compose([
    A.Normalize()
])
# Blur_dataset 설정
blur_dataset = torchvision.datasets.ImageFolder(root='path/to/Blur_dataset', transform=train_transforms)

# Calculate the size of the training and validation sets
total_size = len(blur_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

# Split the dataset into training and validation sets
train_dataset, val_dataset = torch.utils.data.random_split(blur_dataset, [train_size, val_size])

# 데이터 로더 설정
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)

data_lists, data_labels = img_encoder.img_gather("blur_dataset")

best_models = []
k_fold_num = 5

if k_fold_num == -1:
    train_lists, valid_lists, train_labels, valid_labels = train_test_split(data_lists, data_labels, train_size=0.8,
                                                                            shuffle=True, random_state=random_seed,
                                                                            stratify=data_labels)

    train_dataset = TrainDataset(file_lists=train_lists, label_lists=train_labels, transforms=train_transforms)
    valid_dataset = TrainDataset(file_lists=valid_lists, label_lists=valid_labels, transforms=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    data_loader = {"train_loader": train_loader, "valid_loader": valid_loader}

    print("No fold training starts ... ")
    train_result, best_model = Train(data_loader)

    best_models.append(best_model)

else:
    skf = StratifiedKFold(n_splits=k_fold_num, random_state=random_seed, shuffle=True)

    print(f"{k_fold_num} fold training starts ... ")
    for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(data_lists, data_labels), 1):
        print(f"- {fold_idx} fold -")
        train_lists, train_labels = data_lists[train_idx], data_labels[train_idx]
        val_lists, valid_labels = data_lists[valid_idx], data_labels[valid_idx]

        train_dataset = TrainDataset(file_lists=train_lists, label_lists=train_labels, transforms=train_transforms)
        val_dataset = TrainDataset(file_lists=val_lists, label_lists=valid_labels, transforms=valid_transforms)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        data_loader = {"train_loader": train_loader, "val_loader": val_loader}

        train_result, best_model = Train(data_loader)

        best_models.append(best_model)

        model_save_path = f"best_model_fold_{fold_idx}.pt"
        torch.save(best_model, model_save_path)
        print(f"Saved model for fold {fold_idx} to {model_save_path}")