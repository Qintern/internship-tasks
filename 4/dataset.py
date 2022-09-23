from pathlib import Path
import random
import PIL.Image
import numpy as np
import torch
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

IMG_SIZE = 224
CSV_FILENAME = 'data.csv'
TRANSFORM_ZEROCENTER_PATH = 'transform_zerocenter.csv'
CSV_FIELDS = ['id', 'img_location', 'mask_location',
              'min_height', 'max_height', 'min_width',
              'max_width']


class SoilErosionDataset(Dataset):
    def __init__(self, transforms=None, train=True, root_dir='./data/', images_dir='images/', masks_dir='masks/'):
        self.transforms = transforms
        self.train = train
        with open(root_dir + CSV_FILENAME) as csv:
            self.df = pd.read_csv(csv)

        # Get mean and std
        zc_df = pd.read_csv(TRANSFORM_ZEROCENTER_PATH)
        self.mean = zc_df['mean'].to_numpy()
        self.std = zc_df['std'].to_numpy()

        # self.img_dir = Path(root_dir + images_dir)
        # self.masks_dir = Path(root_dir + masks_dir)

    def __getitem__(self, idx):
        # # Load all
        # images = sorted(self.img_dir.glob('*.png'))
        # masks = sorted(self.img_dir.glob('*.png'))

        # Locate files
        img_path = Path(self.df['img_location'][idx])
        mask_path = Path(self.df['mask_location'][idx])

        # Open files
        # img = PIL.Image.open(str(img_path)).convert('RGB')
        # mask = PIL.Image.open(str(mask_path)).convert('P')
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img / 255.0).astype('float32')
        mask = (cv2.imread(str(mask_path), 0) / 255.0).astype('float32')
        # print(img.dtype, mask.dtype)

        # Apply transforms
        if self.transforms is not None:
            img = self.transforms(img)
            # img = TF.normalize(img, mean=self.mean, std=self.std)
            mask = self.transforms(mask)

        return img, mask

        # return {
        #     'image': torch.as_tensor(img.copy()).float().contiguous(),
        #     'mask': torch.as_tensor(mask.copy()).long().contiguous()
        # }

    def __len__(self):
        return len(self.df)

    def transform(self, image, mask):
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random resized crop
        if random.random() > 0.5:
            new_size = random.uniform(0.8, 1.0)
            new_size = int(IMG_SIZE * new_size)
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(new_size, new_size))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            resize = transforms.Resize(size=(IMG_SIZE, IMG_SIZE))
            image = resize(image)
            mask = resize(mask)

        # Random Rotation
        if random.random() > 0.5:
            random_angle = random.randint(-180, 180)
            image = TF.rotate(image, random_angle)
            mask = TF.rotate(mask, random_angle)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask


if __name__ == "__main__":
    dataset = SoilErosionDataset()
    print(len(dataset))
    print(dataset[0])
