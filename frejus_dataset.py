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

def get_datasets(years):
    """
    
    Loads the data (rasters and year vector) as pytorch tensor from the frejus dataset
    
    """
    
    # loading the files
    our_dir = "data/np_data/"
    list_files = os.listdir(our_dir)
    list_files = [our_dir+file for file in list_files]
    
    # sorting the names to have similar order
    list_files.sort(reverse=True)
    
    # storing our rasters per year in a dictionary
    data = {}
    for year in years:
        data[year] = []
    
    # loading the list for the ground truth
    gt_year = []
    
    # loading the rasters
    for year in data:
        for file in list_files:
            if file[13:17] == year in file and "gt" not in file:
                data[year].append(load(file))
                
    # loading the ground truth (corresponding dates)
    for year in data:
        for file in list_files:
            if file[13:17] == year in file and "gt" in file:
                gt_year += [load(file)]
    
    # dict to store our GT rasters (change and classes)
    gt_change ={}
    for year in years:
        gt_change[year] = []
    
    # getting the list of the files
    our_dir = "data/GT_np/"
    list_files_gt = os.listdir(our_dir)
    list_files_gt = [our_dir+file for file in list_files_gt]
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
    data_list = []
    
    for year in data:
        
        data_list += data[year]
    
    # loading the torch data without batch
    datasets = fun.train_val_dataset(data_list, gt_year)
    
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
    
    # as we don't need  evaluation data, we merge the two datasets
    for i in range(len(datasets["train"])):
       train_data.append([datasets["train"][i], datasets["gt_train"][i]])
       
    
    return train_data, gt_change, data
    