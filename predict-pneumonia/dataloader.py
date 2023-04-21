from torch.utils.data import DataLoader
from XRayDataset import XRayDataset
import torchvision.transforms as T

resize = T.Compose([
            T.ToPILImage(), 
            T.Resize((int(968.07), int(968.07/0.7146))), # Resize the image to match mean ratio
            T.ToTensor()
        ])
 
transforms = T.Compose([
            T.ToPILImage(), 
            T.RandomAdjustSharpness(sharpness_factor=10),
            T.ColorJitter(brightness=.5, hue=.3),
            T.Resize((int(968.07), int(968.07/0.7146))), # Resize the image to match mean ratio
            T.ToTensor()
        ])

training_data = XRayDataset("/home/josemaria/30254-pneumoniacs/predict-pneumonia/data/csv_files/train.csv", transforms)
val_data = XRayDataset("/home/josemaria/30254-pneumoniacs/predict-pneumonia/data/csv_files/val.csv", resize)
test_data = XRayDataset("/home/josemaria/30254-pneumoniacs/predict-pneumonia/data/csv_files/test.csv", resize)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)