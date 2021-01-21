# Project hiatus
# various functions for visulization, formatting, etc.
# file with all the functions to charge, format and visualize
# 13/11/2020
# Cédric BARON

# importing libraries
import json
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from rasterio.plot import show
from rasterio.mask import mask
from collections import Counter
from sklearn import metrics
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import matplotlib.patches as mpatches
import random
import argparse
# this is used to load the arguments from the model
from argparse import Namespace

# this is used for the visualize function
from mpl_toolkits.mplot3d import Axes3D

import model as mod

def getFeatures(gdf):
    """
    param: a geopanda dataframe
    Function to parse features from GeoDataFrame in such a manner that rasterio wants them
    """
    
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def nn_interpolate(A, new_size):
    """
    Performs a linear interpolation
    args: A is a numpy matrix, new size is a list
    """
    # get sizes
    old_size = A.shape
    
    # calculate row and column ratios
    row_ratio, col_ratio = new_size[0]/old_size[0], new_size[1]/old_size[1]
    
    # define new pixel row position i
    new_row_positions = np.array(range(new_size[0]))+1
    new_col_positions = np.array(range(new_size[1]))+1
    
    # normalize new row and col positions by ratios
    new_row_positions = new_row_positions / row_ratio
    new_col_positions = new_col_positions / col_ratio
    
    # apply ceil to normalized new row and col positions
    new_row_positions = np.ceil(new_row_positions)
    new_col_positions = np.ceil(new_col_positions)
    
    # find how many times to repeat each element
    row_repeats = np.array(list(Counter(new_row_positions).values()))
    col_repeats = np.array(list(Counter(new_col_positions).values()))
    
    # perform column-wise interpolation on the columns of the matrix
    row_matrix = np.dstack([np.repeat(A[:, i], row_repeats) 
                            for i in range(old_size[1])])[0]
    
    # perform column-wise interpolation on the columns of the matrix
    nrow, ncol = row_matrix.shape
    final_matrix = np.stack([np.repeat(row_matrix[i, :], col_repeats)
                             for i in range(nrow)])

    return final_matrix


def convert_binary(values, thresh):
    """
    With a given thresholds outputs a vector with binary 0/1
    values
    args: values as a matrix or a vector, thresh as an int or float
    
    """
    
    # loading the numpy array
    data = np.array(values)
    binary_vect = np.zeros(data.shape)
        
    ## converting into binary data
    bool_mat = data > thresh
  
    # converting into numpy
    binary_vect[bool_mat] = 1
    
    return binary_vect


def regrid(data, out_x, out_y, interp_method="linear"):
    """
    param: numpy array, number of columns, number of rows
    fun: function to interpolate a raster
    
    """
    
    m = max(data.shape[0], data.shape[1])
    y = np.linspace(0, 1.0/m, data.shape[0])
    x = np.linspace(0, 1.0/m, data.shape[1])
    interpolating_function = RegularGridInterpolator((y, x), data, method=interp_method)
    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, out_y), np.linspace(0, 1.0/m, out_x))
    
    # reprojects the data
    return interpolating_function((xv, yv))


## variables for the visualize function
# creating the x and y values for the 3d plot
a = np.arange(128)
a.reshape((128,1))
b = np.flip(np.arange(128))
y = np.arange(128)
y.reshape((128,1))
x = np.flip(np.arange(128))

# stacking up the values
for i in range(127):
    y = np.column_stack( [ y , a] )
    x = np.row_stack([x, b])



def visualize(raster, third_dim=True, defiance=False):
    """
    param: a raster 2*128*128, with mns and radiometry
    fun: visualize a given raster in two dimensions and in 3d for altitude
    """
    if defiance:
        # creating axes and figures
        fig, ((mns, col), (defi, _)) = plt.subplots(2, 2, figsize=(14, 14)) # Create one plot with figure size 10 by 10
        
        # setting the title
        mns.set_title("mns")
        col.set_title("color")
        defi.set_title("defiance mns")
        mns.axis("off")
        col.axis("off")
        defi.axis("off")
        
        # showing the data
        mns = mns.imshow(raster[0,:,:], vmin=-1.5, vmax=3)
        col = col.imshow(raster[1,:,:], cmap="gray")
        defi = defi.imshow(raster[2,:,:], cmap="hot")#, vmin=0, vmax=2)
        plt.axis("off")
        plt.show()
    
    else:
        # creating axes and figures
        fig, (mns, col) = plt.subplots(1, 2, figsize=(14, 14)) # Create one plot with figure size 10 by 10
        
        # setting the title
        mns.set_title("mns")
        col.set_title("color")
        mns.axis("off")
        col.axis("off")
        
        # showing the data
        mns = mns.imshow(raster[0,:,:], vmin=-1.5, vmax=3)
        
        col = col.imshow(raster[1,:,:], cmap="gray")
        
        
        plt.show()
    
    if third_dim:
        # visualizing in 3d
        ax1 = plt.axes(projection='3d')
        
        # Data for a three-dimensional line
        zline = np.arange(raster[0].min(), raster[0].max(), step=128)
        xline = np.arange(128)
        yline = np.arange(128)
        ax1.plot3D(xline, yline, zline, 'gray')
        
        # Data for three-dimensional scattered points
        zdata = raster[0].flatten()
        xdata = x.flatten()
        ydata = y.flatten()
        ax1.view_init(50, 35)
        ax1.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
        plt.axis("off")
        plt.show()
    

def view_embeddings(fmap, ax = None, show=False):
    """
    param: a tensor, output of a network layer, and an ax plt object for subplotting
    fun: visualizes the embedding using PCA
    """
    
    # if no axes for the plot is specified
    if ax== None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, aspect='equal')
    
    # extracting dimensions of the tensor, nb channels and width/height
    fmap_dim = fmap.shape[1]
    n_pix = fmap.shape[2]
    
    #we use a pca to project the embeddings to a RGB space
    pca = PCA(n_components=3)
    
    pca.fit(np.eye(fmap_dim))
    
    # we need to adapt dimension and memory allocation to CPU
    # transpose makes a matrix transposition
    fmap_ = fmap.cpu().detach().numpy().squeeze().reshape((fmap_dim, n_pix * n_pix)).transpose(1,0)
    # generates the pca data, with 3 components
    color = pca.transform(fmap_)
    
    #we normalize for visibility
    color = np.maximum(np.minimum(((color - color.mean(1, keepdims = True) +0.5) / (2 * color.std(1, keepdims = True))), 1), 0)
    color = color.reshape((n_pix, n_pix,3), order= 'C')
    ax.imshow(color)
    
    # showing the plot
    if show:
        plt.show()
    
    plt.axis('off')
    

def view_u(train, trained_model, args, tile_index = None):
    """
    param: datasets, index of the raster, AE model
    fun: runs the model on the data and visualize various embeddings inside
         the model
    """
    
    # loading the data and reshaping it for prediction
    input = train[tile_index]
    
    # converting to adequate format
    try:
        input = torch_raster((input[None,:,:,:]))
    except:
        input = (input[None,:,:,:]).cuda().float()

    # loading the encoder
    model = trained_model.encoder
    
    # load altitude and reshape it
    alt = input[:,0,:,:][:,None,:,:]
    
    # load rad and reshape it
    rad = input[:,1,:,:][:,None,:,:]
    
    ## running the model
    # encoder alt
    a1 = model.sc2_mns(model.c1_mns(alt))
    #level 2
    a3= model.sc4_mns(model.c3_mns(a1))
    
    #encoder
    #level 1
    x1 = model.sc2_rad(model.c1_rad(rad))
    #level 2
    if args.data_fusion:
        x2= model.c3(x1 + a1)
        
        # extra layer
        x3 = model.sc4(x2)
        
        #level 3
        x4 = model.c5(x3 + a3)
    else:
        x2= model.c3(x1)
        
        # extra layer
        x3 = model.sc4(x2)
        
        #level 3
        x4 = model.c5(x3)
    
    #decoder
    model = trained_model.decoder
    #level 2
    y4 = model.t1(model.c6(x4))
    y3 = model.c8(model.c7(y4))
    
    #level 1
    y2 = model.t2(y3)
    y1 = model.c10(model.c9(y2))
    
    #output        
    print(input.shape)
    
    ## show input
    # creating axes and figures
    fig, (mns, col) = plt.subplots(1, 2, figsize=(14, 14)) # Create one plot with figure size 10 by 10
    # setting the title
    mns.set_title("mns")
    col.set_title("color")
    mns.axis("off")
    col.axis("off")
    
    # plotting the data
    mns = mns.imshow(numpy_raster(input[:,0,:,:]), vmin=-1.5, vmax=3)
    col = col.imshow(numpy_raster(input[:,1,:,:]), cmap="gray")
    plt.show()
    
    ## show various embeddings in the model
    fig = plt.figure(figsize=(25, 10))
    ax = fig.add_subplot(3, 7, 2, aspect=1)
    ax.set(title='x1 : %d x %d x %d' %(x1.shape[1:]))
    view_embeddings(x1, ax)
    ax = fig.add_subplot(3, 7, 9, aspect=1)
    ax.set(title='x2 : %d x %d x %d' %(x2.shape[1:]))
    view_embeddings(x2, ax)
    ax = fig.add_subplot(3, 7, 10, aspect=1)
    ax.set(title='x3 : %d x %d x %d' %(x3.shape[1:]))
    view_embeddings(x3, ax)
    ax = fig.add_subplot(3, 7, 17, aspect=1)
    ax.set(title='x4 : %d x %d x %d' %(x4.shape[1:]))
    
    if args.split:
        view_embeddings(x4[:,:args.nb_channels_split,:,:], ax)
        ax = fig.add_subplot(3, 7, 11, aspect=1)
        ax.set(title='y4 : %d x %d x %d' %(y4.shape[1:]))
        
    else:
        view_embeddings(x4, ax)
        ax = fig.add_subplot(3, 7, 11, aspect=1)
        ax.set(title='y4 : %d x %d x %d' %(y4.shape[1:]))
        
    view_embeddings(y4, ax)
    ax = fig.add_subplot(3, 7, 12, aspect=1)
    ax.set(title='y3 : %d x %d x %d' %(y3.shape[1:]))
    view_embeddings(y3, ax)
    ax = fig.add_subplot(3, 7, 5, aspect=1)
    ax.set(title='y2 : %d x %d x %d' %(y2.shape[1:]))
    view_embeddings(y2, ax)
    ax = fig.add_subplot(3, 7, 6, aspect=1)
    ax.set(title='y1 : %d x %d x %d' %(y1.shape[1:]))
    view_embeddings(y1, ax)


def change_detection(rast1, rast2, trained_model, args, gts = False, visualization=False, threshold=5):
  """
  param: two rasters of dims 1*2*128*128, our neural network model
  fun: outputs a change detection map based on two bi-temporal rasters
  """
  
  # ============rast1===========
  input = torch_raster(rast1)
  
  # load rad and reshape it
  rad1 = input[:,1,:,:][:,None,:,:]
  alt1 = input[:,0,:,:][:,None,:,:]
  
  # loading the encoder
  trained_model = trained_model.encoder
  
  if args.split:
      code_rast1 = trained_model(input, args)[:,:args.nb_channels_split,:,:]
  else:
      code_rast1 = trained_model(input, args)
  
  # ============rast2===========
  input = torch_raster(rast2)
  
  # load rad and reshape it
  rad2 = input[:,1,:,:][:,None,:,:]
  alt2 = input[:,0,:,:][:,None,:,:]

  #level 3
  if args.split:
      code_rast2 = trained_model(input, args)[:,:args.nb_channels_split,:,:]
  else:
      code_rast2 = trained_model(input, args)
      
  # ============cmap===========
  # difference matrix on the code
  CD_code = (code_rast1 - code_rast2) ** 2
  CD_code = torch.mean(CD_code, dim=1)
  CD_code = CD_code ** 0.5
  
  # values below the threshold are converted to zero
  CD_code_cl = CD_code * (CD_code > threshold).float()
  
  # converting into numpy
  CD_code_cl = numpy_raster(CD_code_cl)
  
  ## changing into a binary map
  # checking values that are not zeros
  non_zero_mat = np.nonzero(CD_code_cl)
  
  # creating the binary change map
  cmap_bin = CD_code_cl.copy()
  cmap_bin[non_zero_mat] = 1

  # visualisation of the rasters and the change map
  if visualization == True:
      
      fig = plt.figure(figsize=(25, 10)) #adapted dimension
      fig.suptitle("Change detection on two rasters)", ha="right",
                   size=20)
      ax = fig.add_subplot(3, 7, 9, aspect=1)
      ax.set(title='Change map: float' )
      ax.imshow(CD_code.cpu().detach().numpy().squeeze(), cmap="hot")
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 2, aspect=1)
      ax.set(title='MNS 1' )
      ax.imshow(alt1.cpu().detach().numpy().squeeze(), vmin=-1, vmax=2)
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 16, aspect=1)
      ax.set(title='MNS 2' )
      ax.imshow(alt2.cpu().detach().numpy().squeeze(), vmin=-1, vmax=2)
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 1, aspect=1)
      ax.set(title='Radiometry 1' )
      ax.imshow(rad1.cpu().numpy().squeeze(), cmap="gray")
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 15, aspect=1)
      ax.set(title='Radiometry 2' )
      ax.imshow(rad2.cpu().numpy().squeeze(), cmap="gray")
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 10, aspect=1)
      ax.set(title='Min value: %1.1f, threshold: %2.1f' % (cmap_bin.min(), threshold))
      ax.imshow(cmap_bin.squeeze())
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 3, aspect=1)
      ax.set(title='Code raster 1' )
      view_embeddings(code_rast1, ax)
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 17, aspect=1)
      ax.set(title='Code raster 2' )
      view_embeddings(code_rast2, ax)
      plt.axis('off')
      
      
      # visualising the ground truth
      if gts:
          
          # sub rasters for mns and radiometry
          diff_mns = ((alt1 - alt2)**2)**0.5
          diff_radio = ((rad1 - rad2)**2)**0.5
          
          # colors for the labels
          colors_cmap = ListedColormap(["black", "green", "red"])
          cmap = ListedColormap(['black','blue','purple','yellow'])
            
          # Define a normalization from values -> colors
          norm = colors.BoundaryNorm([0, 1, 2, 3, 4], 5)
          norm_cmap = colors.BoundaryNorm([-1, 0, 1, 2], 4)

          
          # loading the gt change map 
          cmap_gt, data_index, pixel_class = binary_map_gt(gts[0][None,:,:], gts[1][None,:,:])
          
          # loading a raster to visualize the change map, -1 is no data
          gt_map = np.zeros(data_index.shape)
          gt_map += -1
          # loading the gt values
          gt_map[data_index] = cmap_gt
          
          # putting first gt
          ax = fig.add_subplot(3, 7, 4, aspect=1)
          ax.set(title='GT raster 1' )
          ax.imshow(gts[0], cmap=cmap, norm=norm, label="test")
          ## making the legend
          # loading unique values
          cols = ['black','blue','purple','yellow']
          labels = ['nodata', 'buildings', 'roads', 'fields']
          # create a patch (proxy artist) for every color 
          patches = [ mpatches.Patch(color=cols[i], label=labels[i]) for i in range(len(labels)) ]
          ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
          plt.axis('off')
          
          ax = fig.add_subplot(3, 7, 18, aspect=1)
          ax.set(title='GT raster 2' )
          ax.imshow(gts[1], cmap=cmap, norm=norm)
          plt.axis('off')
          
          ax = fig.add_subplot(3, 7, 11, aspect=1)
          ax.set(title='GT cmap, Nodata is -1' )
          ax.imshow(gt_map, cmap=colors_cmap, norm=norm_cmap)
          
          ## making the legend
          # loading unique values
          cols = ['black','green','red']
          labels = ['nodata', 'nochange', 'change']
          # create a patch for every color 
          patches = [ mpatches.Patch(color=cols[i], label=labels[i]) for i in range(len(labels)) ]
          ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
          
          plt.axis('off')
          
          
          # we make the roc analysis if there is relevant GT data
          # in case there misses change or no change pixels there will be an exception
          try:
              
              ## calcuating the roc
              # getting the difference raster to same dimensions
              CD_code = CD_code.detach().cpu().numpy()
              pred_map = regrid(CD_code.squeeze().reshape(CD_code.shape[1:]), 128, 128, "nearest")
              
              # loading the predicted values
              pred_change = pred_map[data_index]
              
              # removing no data values
              diff_mns = diff_mns.detach().cpu().numpy().squeeze()[data_index]
              diff_radio = diff_radio.detach().cpu().numpy().squeeze()[data_index]
              
              ## getting roc for the baseline
              fpr_alt, tpr_alt, thresholds = metrics.roc_curve(cmap_gt, diff_mns)
              fpr_rad, tpr_rad, thresholds = metrics.roc_curve(cmap_gt, diff_radio)
              
              # getting roc values
              fpr, tpr, thresholds = metrics.roc_curve(cmap_gt, pred_change)
              auc = metrics.roc_auc_score(cmap_gt, pred_change)
              
              # plotting
              ax = fig.add_subplot(3, 7, 8, aspect=1)
              ax.set(title='ROC curve, AUC: %1.2f' % (auc))
              ax.plot(fpr, tpr, linestyle='--', label="model")
              ax.plot(fpr_alt, tpr_alt, linestyle=':', label="mns")
              ax.plot(fpr_rad, tpr_rad, linestyle='-', label="radio")
              ax.legend()
             
          except:
              None
          
  return cmap_bin, CD_code, code_rast1, code_rast2


def clipping_rasters(dict_rasters, boxes):
    """
    params: dictionary with years as key and corresponding rasters as values, 
            boxes as a list of dictionaries containing bounding boxes
    fun: outputs a dictionary with years as keys and as values the clipped
         rasters
    """
    
    # creating a dict that will store the clipped rasters
    rasters_clipped = {}
    
    for year in dict_rasters:
        
        # creating our year index for the adversarial part
        rasters_clipped[year] = []
        
    for our_box in boxes:
        
        # list of rasters matching the box
        rasters_box = []
        
        for year in dict_rasters:
            
            for rast in dict_rasters[year]:
                # we place an exception in case the mask isn't working (box outside the raster)
                try:
                    # cropping the raster
                    out_img, out_transform = mask(dataset=rast, all_touched=True,
                                                  shapes=our_box, crop=True)
                    
                    # storing the raster in our list
                    # removing rasters with too many zeros
                    values = out_img.flatten()
                    nb_zeroes = np.count_nonzero(values == 0)
                    
                    # if there are two many zeros we don't select the raster
                    if nb_zeroes > len(values)/5 :
                        None
                    # we regrid the rasters to 128*128 pixels
                    else:
                        resh_rast = regrid(out_img.reshape(out_img.shape[1:]), 128, 128)
                        rasters_box.append(resh_rast)
                
                except:
                    None
        

        # storing the raster per year
        i = 0
        
        for year in dict_rasters:
            
            # appending the rasters into a year index
            rasters_clipped[year].append([rasters_box[i], rasters_box[i+1]])
            i += 2
    
    return rasters_clipped


def pca_visualization(code):
    """
    makes a pca visualization (three components) on the code
    """
    
    # converting into numpy and flattening
    code = numpy_raster(code).reshape(16,32*32)
    
    # performing the PCA
    data_pca = PCA(n_components=3)
    data_pca.fit(code)
    
    # reshaping for visualisation
    data_pc = data_pca.components_.reshape(3,32,32)
    
    # visualisation
    show(data_pc[0,:,:])
    show(data_pc[1,:,:])
    show(data_pc[2,:,:])
    
    
def binary_map_gt(rast1, rast2):
    """
    returns a binary map from the ground truth with two dates
    
    """
    
    # loading the rasters
    gt1 = rast1[0,:,:]
    gt2 = rast2[0,:,:]
    
    ## getting the change map
    # getting the nodata matrix
    data_index = gt1 != 0
    nodata2 = gt2 == 0
    data_index[nodata2] = False
    
    # loading gts with masks
    gt1_cl = gt1[data_index]
    gt2_cl = gt2[data_index]
    
    # making a binary map
    cmap_gt = gt1_cl.copy()
    cmap_gt_bol_change = gt1_cl != gt2_cl
    cmap_gt_bol_nochange = gt1_cl == gt2_cl
    cmap_gt[cmap_gt_bol_change] = 1
    cmap_gt[cmap_gt_bol_nochange] = 0
    
    # original class
    classes = gt2_cl
    
    
    return cmap_gt, data_index, classes
    

def torch_raster(raster):
    """
    function that adapts a raster for the model, change to torch tensor, on cuda,
    float
    """
    
    # converting the data
    result = torch.from_numpy(raster).cuda().float()
    
    return result

def numpy_raster(raster):
    """
    function that adapts a raster for the model, change to torch tensor, on cuda,
    float
    """
    
    # converting the result
    result = raster.detach().cpu().numpy().squeeze()
    
    return result


def prepare_nmi(list_rasters, discrete=False):
    """
    Function to prepare the data for the normalized mutual information
    Arguments are a list of rasters and a boolean in case of discrete (labels) data
    """
    
    # list to store the rasters
    reshap_rasts = []
    
    # reshaping and loading in the list
    for rast in list_rasters:
        
        # reshaping
        rast_resh =  regrid(rast.reshape(rast.shape), 32, 32, "nearest")
        
        # converting again back to integers (reshaping generates floats)
        if discrete:
            rast_resh = np.rint(rast_resh)
        
        # storing into our list
        reshap_rasts.append(rast_resh)
        
    # stacking into one matrix
    matrix_labels = np.stack(reshap_rasts, axis=0)
    
    # reshaping
    matrix_flat = matrix_labels.reshape((len(reshap_rasts)*32*32))
    
    return matrix_flat


def train_val_dataset(dataset, gt, val_split=0.25):
        """
        param: list of rasters as numpy objects and percentage of test data
        fun: outputs a dictionary with training and test data
        """
        
        # getting ids for training and validation sets
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
        
        # subsetting into training and validation, storing into a dictionary
        datasets = {}
        datasets['train'] = Subset(dataset, train_idx)
        datasets['val'] = Subset(dataset, val_idx)
        
        # subsetting the groundtruth for the adversarial part
        datasets['gt_train'] = Subset(gt, train_idx)
        datasets['gt_val'] = Subset(gt, val_idx)
        
        return datasets


def visu_result_model(losses):
    """
    
    shows various graphs with the losses from our model
    
    """
    
    # graphs of different losses
    plt.title('loss per number of epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(len(losses["tot"])), losses["tot"])
    plt.show()
  
    plt.title('loss mns per number of epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(len(losses["mns"])), losses["mns"])
    plt.show()
  
    plt.title('loss rad per number of epochs')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(range(len(losses["alt"])), losses["alt"])
    plt.show()
  
    plt.title('accuracy of the discriminator')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(range(len(losses["accu"])), losses["accu"])
    plt.show()
  
    plt.title('AUC')
    plt.xlabel('epoch')
    plt.ylabel('auc')
    plt.plot(range(len(losses["auc"])), losses["auc"])
    plt.show()
  
    print("AUC on average is {} ".format(np.mean(losses["auc"])))
    print("AUC sd is {} ".format(np.std(losses["auc"])))
   
    return None


def reject_outliers(data, m = 3.):
    """
    A function that removes outliers above a certain number of standard deviations
    from the median.
    """
    
    # getting only positive values
    d = np.abs(data - np.median(data))
    
    # getting the median
    mdev = np.median(d)
    
    # normalizing
    s = d/mdev if mdev else 0.
    
    # removing data above a certain number of standard deviations from the median
    return data[s<m]


def get_min(matrix, i=1000):
    """
    Function to compute the ith value starting from the minimum
    """
    
    # making a vector and sorting the values
    mat_flat = matrix.copy()
    mat_flat = mat_flat.flatten()
    mat_flat = np.sort(mat_flat)
    
    # extract the ith value s, starting from the lowest
    minimum = mat_flat[i]
    
    return minimum


def set_seed(seed, cuda=True):
        """ 
        Sets seeds in all frameworks
        """
        
        # setting the seed for various libraries
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if cuda: 
            torch.cuda.manual_seed(seed)  
            
            
def correlation(x, y):
    """
    A function to compute the correlation coefficient
    """
    
    # computing the covariance
    cov = torch.sum((x - x.mean()) * (y - y.mean()))
    
    # computing the standard deviations
    std_x = torch.sqrt(torch.sum((x - torch.mean(x))**2))
    std_y =  torch.sqrt(torch.sum((y - torch.mean(y))**2))
    
    # computing r
    r = cov / (std_x * std_y)
    
    
    return r
    
    
def arguments_parser(parser):
    """
    Loads the arguments of a parser
    """
    # Optimization arguments
    parser.add_argument('--lr', default=0, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default=0, help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--epochs', default=0, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=0, type=int, help='Batch size')
    parser.add_argument('--optim', default=0, help='Optimizer: sgd|adam')
    parser.add_argument('--grad_clip', default=0, type=float, help='Element-wise clipping of gradient. If 0, does not clip')
    
    # Learning process arguments
    parser.add_argument('--cuda', default=0, type=int, help='Bool, use cuda')
    parser.add_argument('--test_nth_epoch', default=0, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--save_nth_epoch', default=0, type=int, help='Save model each n-th epoch during training')

    # Dataset
    parser.add_argument('--dataset', default='frejus_dataset', help='Dataset name: frejus_dataset')
    
    # Model
    parser.add_argument('--seed', default=0, type=int, help='Seed for random initialisation')
    parser.add_argument('--save', default=0, type=int, help='Seed for random initialisation')
    parser.add_argument('--data_fusion', default=0, help='Including data fusion')
    parser.add_argument('--adversarial', default=0, help='Making the model adversarial')
    parser.add_argument('--defiance', default=0, help='Including defiance')
    parser.add_argument('--split', default=0, help='Making a split on the code')
    parser.add_argument('--auto_encod', default=0, help='Activating the auto-encoder')
    
    # Encoder
    parser.add_argument('--conv_width', default=0, help='Layers size')
    
    # Decoder
    parser.add_argument('--dconv_width', default=0, help='Layers size')
    
    # defiance
    parser.add_argument('--def_width', default=0, help='Layers size')
    
    # Discriminator
    parser.add_argument('--nb_channels_split', default=0, type=int, help='Number of channels for the input to the discriminator')
    parser.add_argument('--disc_width', default=0, help='Layers size')
    parser.add_argument('--nb_trains_discr', default=0, type=int, help='Number of times the discriminator is trained compared to the autoencoder')
    parser.add_argument('--disc_loss_weight', default=0, type=float, help='Weight applied on the adversarial loss with full model')
    parser.add_argument('--opti_adversarial_encoder', default=0, help='Trains the encoder weights')
    
    return parser
    
    
def load_model(path_model, path_args):
    
    ## get the arguments from the model
    # getting the arguments as a string from the text file
    file1 = open(path_args, 'r') 
    args_str = file1.read()
    file1.close()
    
    # creating the parser and the arguments
    parser = argparse.ArgumentParser()
    parser = arguments_parser(parser)
    args = parser.parse_args()
    
    # changing the arguments values
    args = parser.parse_args(namespace=eval(args_str))
        
    
    #initialize the models
    encoder = mod.Encoder(args.conv_width, args)
    decoder = mod.Decoder(args.conv_width, args.dconv_width, args)
    discr = mod.Discriminator(args)
    trained_model = mod.AdversarialAutoEncoder(encoder, decoder, discr, 0)
    trained_model.load_state_dict(torch.load(path_model))
    trained_model.eval()
    
    return trained_model, args


def prepare_codes_metrics(gt_change, args, trained_model):
    
    ## extracting the codes
    # load list of codes
    list_codes = []
    
    # convert the rasters into codes
    for year in gt_change:
        
        if args.split:
            list_codes += [trained_model.encoder(torch_raster(rast[None,1:,:,:]), args)[:,:args.nb_channels_split,:,:] for rast in gt_change[year]]
        else:
            list_codes += [trained_model.encoder(torch_raster(rast[None,1:,:,:]), args) for rast in gt_change[year]]
        
    # convert them back to numpy matrixes
    np_codes = [rast.detach().cpu().numpy() for rast in list_codes]
        
    # stacking into one matrix
    matrix_codes = np.stack(np_codes, axis=0)
    matrix_codes = matrix_codes.squeeze()
    
    # reshaping
    if args.split:
        flat_codes = matrix_codes.transpose(0,2,3,1).reshape((matrix_codes.shape[0]*32*32, args.nb_channels_split))
    else:
        flat_codes = matrix_codes.transpose(0,2,3,1).reshape((matrix_codes.shape[0]*32*32, 32))
        
    ## extracting the labels
    # load list of labels
    list_labels = []
    
    # loading the labels
    for year in gt_change:
        list_labels += [rast[0,:,:] for rast in gt_change[year]]
        
    # transposing into one matrix
    flat_labels = prepare_nmi(list_labels, discrete=True)
    
    ## removing the no data values
    # getting the nodata matrix
    data_index = flat_labels != 0
    
    # applying the mask
    labels_clean = flat_labels[data_index]
    codes_clean = flat_codes[data_index,:]
    
    
    return codes_clean, labels_clean


def prepare_data_metrics(gt_change, index_data):
    
    
    ## extracting the altitude
    # load list of mns
    list_data = []
    
    # loading the mns
    for year in gt_change:
        list_data += [rast[index_data,:,:] for rast in gt_change[year]]
        
    # reshape to have one single matrix
    flat_data = prepare_nmi(list_data)
    
    ## extracting the labels
    # load list of labels
    list_labels = []
    
    # loading the labels
    for year in gt_change:
        list_labels += [rast[0,:,:] for rast in gt_change[year]]
        
    # transposing into one matrix
    flat_labels = prepare_nmi(list_labels, discrete=True)
    
    ## removing the no data values
    # getting the nodata matrix
    data_index = flat_labels != 0
    
    # applying the mask
    data_clean = flat_data[data_index]
    
    return data_clean


