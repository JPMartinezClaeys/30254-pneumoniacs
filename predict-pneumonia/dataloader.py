from torch.utils.data import DataLoader
from XRayDataset import XRayDataset
import torchvision.transforms as T

resize = T.Compose([
            T.ToPILImage(), 
            T.Resize((int(968.07), int(968.07/0.7146))), # Resize the image to match mean ratio
            T.ToTensor(), 
            T.Normalize(mean=[0.7146]*3, std=[0.1185]*3) # Normalize the image
        ])
 
transforms = T.Compose([
            T.ToPILImage(), 
            T.RandomAdjustSharpness(sharpness_factor=0.5),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            T.Resize((int(968.07), int(968.07/0.7146))), # Resize the image to match mean ratio
            T.ToTensor(), 
            T.Normalize(mean=[0.7146]*3, std=[0.1185]*3)
        ])

training_data = XRayDataset("/home/josemaria/30254-pneumoniacs/predict-pneumonia/data/csv_files/train.csv", transforms)
val_data = XRayDataset("/home/josemaria/30254-pneumoniacs/predict-pneumonia/data/csv_files/val.csv", resize)
test_data = XRayDataset("/home/josemaria/30254-pneumoniacs/predict-pneumonia/data/csv_files/test.csv", resize)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)