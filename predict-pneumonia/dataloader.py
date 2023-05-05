from torch.utils.data import DataLoader
from XRayDataset import XRayDataset
import torchvision.transforms as T
import os

resize = T.Compose([
            T.ToPILImage(), 
            T.Resize((int(888.0), int(888.0/0.71))), # Resize the image to match median ratio using median length, we can try later with smaller length 
            T.ToTensor()
        ])
 
transforms = T.Compose([
            T.ToPILImage(), 
            T.RandomAdjustSharpness(sharpness_factor=10),
            T.ColorJitter(brightness=.5, hue=.3),
            T.Resize((int(888.0), int(888.0/0.71))), # Resize the image to match median ratio using median length 
            T.ToTensor()
        ])

#Read data and pass to dataloader
current_dir = os.getcwd()
trainig_rel_path = "data/csv_files/train.csv"
train_path = os.path.join(current_dir,trainig_rel_path)
val_rel_path = "data/csv_files/val.csv"
val_path = os.path.join(current_dir,val_rel_path)
test_rel_path = "data/csv_files/test.csv"
test_path = os.path.join(current_dir,test_rel_path)

training_data = XRayDataset(train_path, transforms)
val_data = XRayDataset(val_path, resize)
test_data = XRayDataset(test_path, resize)

#Discussion with Ian said to start with small batches (2,4)
#Medium post said do powers of 2 starting with 16 
#I'll go with middle point and do batch = 8
#https://medium.com/data-science-365/determining-the-right-batch-size-for-a-neural-network-to-get-better-and-faster-results-7a8662830f15

train_dataloader = DataLoader(training_data, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=8, shuffle=True)