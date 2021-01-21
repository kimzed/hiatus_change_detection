# Project hiatus
# script to evaluate our model
# 26/11/2020
# Cédric BARON

import pandas as pd
from sklearn import metrics
import numpy as np

from argparse import Namespace
import argparse
import torch

# importing our functions
import utils as fun
import model as mod
import metrics as fun_metrics

def generate_prediction_model(list_rast_gt, model, args):
    """
    Function to generate the change raster from the model
    args: the ground truth rasters as a dictionary (per year), with mns, rad and
          labels ; the model and its arguments (parameters of the model)
    outputs the codes, the binary change maps (gt) and the classes
          
    """
    
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


def generate_prediction_baseline(list_rast_gt):
    """
    Function to output the change map for the baseline
    outputs the float change map (baseline) for the mns, the radiometry and the
    binary map (ground truth)
    """
    
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

def evaluate_model(model, gt_change):
    """
    
    
    
    """
    
    ## get the arguments from the model
    # getting the arguments as a string from the text file
    file1 = open('evaluation_models/'+model+'.txt', 'r') 
    args_str = file1.read()
    file1.close()
    
    # creating the parser and the arguments
    parser = argparse.ArgumentParser()
    parser = fun.arguments_parser(parser)
    args = parser.parse_args()
    
    # changing the arguments values
    args = parser.parse_args(namespace=eval(args_str))
        
    
    #initialize the models
    encoder = mod.Encoder(args.conv_width, args)
    decoder = mod.Decoder(args.conv_width, args.dconv_width, args)
    discr = mod.Discriminator(args)
    trained_model = mod.AdversarialAutoEncoder(encoder, decoder, discr, 0)
    trained_model.load_state_dict(torch.load("evaluation_models/"+model))
    trained_model.eval()
    
    print(
    """
    Checking performance on ground truth change maps
    We output the code subtraction with the model and on the baseline (simple
    rasters subtraction)
    """)
    
    # evaluating the model
    pred, y, classes = generate_prediction_model(gt_change, trained_model,
                                                 args)
    
    # ROC
    print("AUC model")
    fun_metrics.visualize_roc(y, pred, return_thresh=False)
    
    print(
    """
    Performing normalized mutual information for continuous variables
    """)
        
    ## extracting the codes
    # load list of codes
    list_codes = []
    
    # convert the rasters into codes
    for year in gt_change:
        
        if args.split:
            list_codes += [trained_model.encoder(fun.torch_raster(rast[None,1:,:,:]), args)[:,:args.nb_channels_split,:,:] for rast in gt_change[year]]
        else:
            list_codes += [trained_model.encoder(fun.torch_raster(rast[None,1:,:,:]), args) for rast in gt_change[year]]
        
    # convert them back to numpy matrixes
    np_codes = [rast.detach().cpu().numpy() for rast in list_codes]
        
    # stacking into one matrix
    matrix_codes = np.stack(np_codes, axis=0)
    matrix_codes = matrix_codes.squeeze()
    
    # reshaping
    if args.split:
        flat_codes = matrix_codes.transpose(0,2,3,1).reshape((matrix_codes.shape[0]*32*32, 4))
    else:
        flat_codes = matrix_codes.transpose(0,2,3,1).reshape((matrix_codes.shape[0]*32*32, 32))
    
    ## extracting the labels
    # load list of labels
    list_labels = []
    
    # loading the labels
    for year in gt_change:
        list_labels += [rast[0,:,:] for rast in gt_change[year]]
        
    # transposing into one matrix
    flat_labels = fun.prepare_nmi(list_labels, discrete=True)
    
    ## removing the no data values
    # getting the nodata matrix
    data_index = flat_labels != 0
    
    # applying the mask
    labels_clean = flat_labels[data_index]
    codes_clean = flat_codes[data_index, :]
    
    ## getting the number of pixels per classes
    nb_build = np.count_nonzero(labels_clean == 1)
    nb_road = np.count_nonzero(labels_clean == 2)
    nb_field = np.count_nonzero(labels_clean == 3)
    nb_classes = (nb_build, nb_road, nb_field)
    
    ## spliting the dataset according to the class
    # loading the data
    buildings_idx = labels_clean == 1
    roads_idx = labels_clean == 2
    fields_idx =  labels_clean == 3
    
    # putting into a list
    classes_idx = [buildings_idx, roads_idx, fields_idx]
    
    # calculating the NMI for the codes
    mi_score = fun_metrics.NMI_continuous_discrete(labels_clean, codes_clean,
                                        nb_classes, [1,2,3], classes_idx)
    print("NMI score for the model is %1.2f" % (mi_score))
    
    
    print(
    """
    Making a linear SVM
    """)
    
    ## linear svm with the model
    conf_mat_model, class_report_model, scores_cv = fun_metrics.svm_accuracy_estimation(codes_clean,
                                                                             labels_clean)
    
    print("Results for the model")
    print("/n")
    print(class_report_model)
    
   
    
    return None
