# Project hiatus
# script to load the frejus dataset
# 12/10/2020
# Cedric BARON

# loading required packages
import torch
from numpy import load
import os

# importing used functions
import utils as fun

def get_datasets(ex_raster=False):
    """
    
    Loads the data (rasters and year vector) as pytorch tensor from the frejus dataset
    And raster can be loaded as an example to visualize evolution of the model
    after every epoch
    
    """
    
    # putting the right directory for the data
    os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection/data/np_data/")
    
    # loadint the files
    list_files = os.listdir()
    
    # sorting the names to have similar order
    list_files.sort(reverse=True)
    
    # storing our rasters per year in a dictionary
    s_rasters_clipped = {"1954":[], "1966":[], "1970":[], "1978":[], "1989":[]}
    
    # loading the list for the ground truth
    gt = []
    
    # loading the rasters
    for year in s_rasters_clipped:
        for file in list_files:
            if file[:4] == year in file and "gt" not in file:
                s_rasters_clipped[year].append(load(file))
                
    # loading the ground truth (corresponding dates)
    for year in s_rasters_clipped:
        for file in list_files:
            if file[:4] == year in file and "gt" in file:
                gt += [load(file)]
    
    # dict to store our GT rasters
    gt_change = {"1954":[], "1966":[], "1970":[], "1978":[], "1989":[]}
    
    # putting the right directory for the data
    os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection/data/GT_np/")
    
    # getting the list of the files
    list_files_gt = os.listdir()
    list_files_gt.sort()
    
    # loading the matrixes in the dict per year
    for file in list_files_gt:
        for year in gt_change:
            if year in file and "class" in file:
                gt_change[year].append(load(file))
          
    """
    
    we now build our dataset as a list of tensors
    
    """
    
    # stacking up the  samples into a list
    m_samples = []
    
    for year in s_rasters_clipped:
        
        m_samples += s_rasters_clipped[year]
    
    
    
    # loading the torch data without batch
    datasets = fun.train_val_dataset(m_samples, gt)
    
    # extracting evals, converting into pytorch tensors
    datasets["val"] = [torch.from_numpy(obs) for obs in datasets["val"]]
    datasets["gt_val"] = [torch.from_numpy(obs) for obs in datasets["gt_val"]]
    
    # extracting only images for the training, converting into tensors
    datasets["train"] = [torch.from_numpy(obs) for obs in datasets["train"]]
    datasets["gt_train"] = [torch.from_numpy(obs) for obs in datasets["gt_train"]]
    
    # merging val and train because we want more samples
    datasets["train"] = datasets["train"] + datasets["val"]
    datasets["gt_train"] = datasets["gt_train"] + datasets["gt_val"]
    
    ## we need to combine images and labels for the discriminator
    train_data = []
    
    for i in range(len(datasets["train"])):
       train_data.append([datasets["train"][i], datasets["gt_train"][i]])
       
    if ex_raster:
        # loading a raster to check the models updates
        ex_raster = fun.torch_raster(s_rasters_clipped["1970"][715][None,:,:,:])
    else:
        ex_raster = None
    
    return train_data, gt_change, s_rasters_clipped, ex_raster
    
    
    
    
    
    
    