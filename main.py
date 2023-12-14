import albumentations as A
from sklearn.model_selection import train_test_split
import img_encoder
import torch
import numpy as np

random_seed = 42

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

num_workers = 2
batch_size = 48

train_transforms = A.Compose([
    A.Rotate(),
    A.HorizontalFlip(),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=-.2, hue=0.2),
    A.Normalize()
])

val_transforms = A.Compose([
    A.Normalize()
])

data_lists, data_labels = img_encoder.img_gather("blur_dataset")

best_models = []
k_fold_num = 5

if k_fold_num == -1:
    train_lists, val_lists, train_labels, val_labels = train_test_split(
        data_lists, data_labels, train_size=0.8, shuffle=True, random_state=random_seed, stratify=data_labels)