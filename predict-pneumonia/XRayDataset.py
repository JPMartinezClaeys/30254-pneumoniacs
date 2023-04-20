import os
import pandas as pd
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.io import read_image

import cv2

class XRayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        You can set your custom dataset to take in more parameters than specified
        here. But, I recommend at least you start with the three I listed here,
        as these are standard

        csv_file (str): file path to the csv file you created /
        df (pandas df): pandas dataframe

        img_dir_path: directory path to your images
        transform: Compose (a PyTorch Class) that strings together several
          transform functions (e.g. data augmentation steps)

        One thing to note -- you technically could implement `transform` within
        the dataset. No one is going to stop you, but you can think of the
        transformations/augmentations you do as a hyperparameter. If you treat
        it as a hyperparameter, you want to be able to experiment with different
        transformations, and therefore, it would make more sense to decide those
        transformations outside the dataset class and pass it to the dataset!
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = self.data['filename']
        self.transform = transform

    def __len__(self):
        """
        Returns: (int) length of your dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Loads and returns your sample (the image and the label) at the
        specified index

        Parameter: idx (int): index of interest

        Returns: image, label
        """
        row = self.data.iloc[idx]
        label = row['label']
        image = cv2.imread(row['filename'])
     
        if self.transform:
            image = self.transform(image)
        if not self.transform:
            image = self.resize(image)
  
        return image, label
    
    def resize(self):
        length = self.data['length'].mean()
        ratio = self.data['ratio'].mean()
        T.Compose([
            T.ToPILImage(), 
            T.Resize((int(length), int(length/ratio))), # Resize the image to match mean ratio
            T.ToTensor(), 
            T.Normalize(mean=[self.data['ratio'].mean()]*3, std=[self.data['ratio'].std()]*3) # Normalize the image
        ])

    def transform(self):
        length = self.data['length'].mean()
        ratio = self.data['ratio'].mean()    
        T.Compose([
            T.ToPILImage(), 
            T.RandomAdjustSharpness(sharpness_factor=0.5),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.Resize((int(length), int(length/ratio))), # Resize the image to match mean ratio
            T.ToTensor(), 
            T.Normalize(mean=[self.data['ratio'].mean()]*3, std=[self.data['ratio'].std()]*3) 
        ])