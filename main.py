import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random
import cv2
import os
import glob

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.autograd import Variable

from sklearn.model_selection import train_test_split
from skimage import metrics

os.makedirs('motion_blurred', exist_ok=True)
src_dir = 'dir'
images = os.listdir(src_dir)
dst_dir = 'motion_blurred'
size = 11
kernel_motion_blur = np.zeros((size, size))
kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
kernel_motion_blur = kernel_motion_blur/size
for i, img in tqdm(enumerate(images), total=len(images)):
    img = cv2.imread(f"{src_dir}/{images[i]}")
    blur = cv2.filter2D(img, -1, kernel_motion_blur)
    cv2.imwrite(f"{dst_dir}/{images[i]}", blur)
print('Done')