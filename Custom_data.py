import cv2
import numpy as np
from urllib.request import urlopen
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from rembg import remove

class CustomDataset(Dataset):
    def __init__(self, urls,classes,patch_size, transform=None):
        self.urls = urls
        self.classes = classes
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        image_path = self.urls[idx]

        image = self.url_to_image(image_path)
        mask = remove(image,only_mask=True)
        res = cv2.bitwise_and(image,image,mask = mask)
        # PIL.Image를 사용하여 Image Load
        image = Image.fromarray(image)
        clss = float(self.classes[idx])

        patches = self.extract_patches(image)

        if self.transform:
            patches = [self.transform(patch) for patch in patches]
        return patches,np.ones(len(patches))*clss

    def extract_patches(self, image):
        width, height = image.size
        patches = []

        for x in range(0, width, self.patch_size):
            for y in range(0, height, self.patch_size):
                patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))
                if np.sum(np.array(patch).flatten() != 0) > self.patch_size*self.patch_size*0.5:
                    patches.append(patch)

        return patches


    def url_to_image(self,url, readFlag=cv2.IMREAD_COLOR):
        # 이미지를 다운로드하고 NumPy 배열로 변환한 다음 읽습니다.
        # OpenCV 형식으로 변환
        resp = urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, readFlag)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지를 반환
        return image