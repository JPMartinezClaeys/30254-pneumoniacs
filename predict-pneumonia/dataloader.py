from torch.utils.data import DataLoader
from .XRayDataset import XRayDataset

training_data = XRayDataset("/home/josemaria/30254-pneumoniacs/predict-pneumonia/data/csv_files/train.csv")
val_data = XRayDataset("/home/josemaria/30254-pneumoniacs/predict-pneumonia/data/csv_files/val.csv")
test_data = XRayDataset("/home/josemaria/30254-pneumoniacs/predict-pneumonia/data/csv_files/test.csv")

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)