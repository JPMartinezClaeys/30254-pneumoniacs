#Import packages
import cv2
#from torch.utils import data
#from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms as T
import pandas as pd
import matplotlib.pyplot as plt
#import pytorch_lightning as pl
import os

#1. Create df with images info

sets = ["test","train","val"]
labels = ["NORMAL","PNEUMONIA"]

current_dir = os.getcwd()

imgs_df = pd.DataFrame(columns = ['filename', 'shape'])
for dataset in sets:
    for label in labels:
        rel_path = "data/{}/{}".format(dataset,label)
        img_directory = os.path.join(current_dir, rel_path)
        for img_file in os.listdir(img_directory):
            img_path = os.path.join(img_directory,img_file)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img_shape = img.shape
            img_l = img_shape[0]
            img_w = img_shape[1]
            img_ratio = round(img_l/img_w,2)
            imgs_df = imgs_df.append({'filename':"{}/{}".format(rel_path,img_file),'shape':img.shape,
                'length':img_l,'width':img_w,'ratio':img_ratio,'label':label,
                'set':dataset},ignore_index=True)

#2.Graph distribution of images
def plot_shape_distrib(shape_col,n_bins=20):
    """ 
    Plot a histogram to visualize distribution of images shapes attributes
    Input:
    - shape_col(str): Column to visualize in histogram
    -n_bins(int): Number of bins for the 
    Return:
        None, visualizes plt histogram on Jupyter Notebook
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    if shape_col != "ratio":
        ax.set_xlabel("{} (pixels)".format(shape_col), fontsize=16)
    else: 
        ax.set_xlabel("ratio (length/width)", fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.set_title("Distribution of images {}s".format(shape_col), fontsize=16)
    ax.hist(imgs_df.loc[:,shape_col], color='pink',edgecolor='black',bins=n_bins)

plot_shape_distrib("width")
plot_shape_distrib("length")
plot_shape_distrib("ratio")

#3. Save csv for each dataset
imgs_df_test = imgs_df.loc[imgs_df.loc[:,"set"]=="test",:] 
imgs_df_train = imgs_df.loc[imgs_df.loc[:,"set"]=="train",:] 
imgs_df_val = imgs_df.loc[imgs_df.loc[:,"set"]=="val",:] 

imgs_df_test.to_csv(os.path.join(current_dir, "data/csv_files/test.csv"), index = False)
imgs_df_train.to_csv(os.path.join(current_dir, "data/csv_files/train.csv"), index = False)
imgs_df_train.to_csv(os.path.join(current_dir, "data/csv_files/val.csv"), index = False)


#4. Count number of observations
len(imgs_df.loc[imgs_df.loc[:,"label"]=="NORMAL",:])
len(imgs_df.loc[imgs_df.loc[:,"label"]=="PNEUMONIA",:])

#Get info by set and label
for dataset in sets:
    for label in labels:
        n_obs = len(imgs_df.loc[(imgs_df.loc[:,"label"]=="{}".format(label)) & (imgs_df.loc[:,"set"]=="{}".format(dataset)),:])
        print("Number of observations in",dataset, "with label", label," = ",n_obs)