# Project hiatus
# script to evaluate our model
# 26/11/2020
# Cédric BARON

import pandas as pd
from sklearn import metrics
import numpy as np

# importing our functions
import utils as fun

def evaluate_model(list_rast_gt, model, args):
    
    # loading lists to store the results
    y = []
    pred = []
    classes = []
    
    # getting th year for the first rasters
    for year1 in list_rast_gt:
        
        # getting the year for the second raster
        for year2 in list_rast_gt:
            
            # checking that both year are not the same
            if year1 != year2 and year2 > year1:
                
                # getting accuracy score on random combinations
                for ind in range(len(list_rast_gt[year2])):
                    
                    # loading the two rasters
                    rast1 = list_rast_gt[year1][ind]
                    rast2 = list_rast_gt[year2][ind]
                    
                    # loading the gt change map and the mask for no data
                    cmap_gt, data_index, pixel_class = fun.binary_map_gt(rast1, rast2)
                    
                    # loading the rasters
                    rast1 = rast1[1:,:,:][None,:,:,:]
                    rast2 = rast2[1:,:,:][None,:,:,:]
                    
                    # computing change raster
                    cmap, dccode, code1, code2 = fun.change_detection(rast1,
                                                                      rast2,
                                                                      model, 
                                                                      args,
                                                                      visualization=False)
                    
                    dccode = dccode.detach().cpu().numpy()
                    
                    # reshaping to original dimensions
                    pred_map = fun.regrid(dccode.reshape(dccode.shape[1:]), 128, 128, "nearest")
                    
                    # removing no data
                    cmap_pred = pred_map.squeeze()[data_index]
                    
                    # storing the results and corresponding classes
                    pred += list(cmap_pred)
                    classes += list(pixel_class)
                    y += list(cmap_gt)
    
    
    return pred, y, classes


def evaluate_baseline(list_rast_gt):
    
    # loading lists to store the results
    y = []
    pred_rad = []
    pred_alt = []
    
    # getting th year for the first rasters
    for year1 in list_rast_gt:
        
        # getting the year for the second raster
        for year2 in list_rast_gt:
            
            # checking that both year are not the same
            if year1 != year2 and year2 > year1:
                
                # getting accuracy score on random combinations
                for ind in range(len(list_rast_gt[year2])):
                    
                    
                    # loading the two rasters
                    rast1 = list_rast_gt[year1][ind]
                    rast2 = list_rast_gt[year2][ind]
                    
                    # loading the gt change map and the mask for no data
                    cmap_gt, data_index, classes = fun.binary_map_gt(rast1, rast2)
                    
                    # loading the rasters
                    rast1 = rast1[1:,:,:][None,:,:,:]
                    rast2 = rast2[1:,:,:][None,:,:,:]
                    
                    # computing change raster
                    sub_alt = ((rast1[:,0,:,:] - rast2[:,0,:,:])**2)**0.5
                    sub_rad = ((rast1[:,1,:,:] - rast2[:,1,:,:])**2)**0.5
                    
                    # removing no data
                    cmap_pred_alt = sub_alt.squeeze()[data_index]
                    cmap_pred_rad = sub_rad.squeeze()[data_index]
                    
                    # storing the results
                    pred_alt += list(cmap_pred_alt)
                    pred_rad += list(cmap_pred_rad)
                    y += list(cmap_gt)
    
    
    
    return pred_alt, pred_rad, y


def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-fpr, index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 