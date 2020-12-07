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
import matplotlib
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# this is used for the visualize function
from mpl_toolkits.mplot3d import Axes3D

import metrics as fun_metrics

def getFeatures(gdf):
    """
    param: a geopanda dataframe
    Function to parse features from GeoDataFrame in such a manner that rasterio wants them
    """
    
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def nn_interpolate(A, new_size):
    """
    Nearest Neighbor Interpolation, Step by Step
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
        fig, ((mns1, col), (def1, def2)) = plt.subplots(2, 2, figsize=(14, 14)) # Create one plot with figure size 10 by 10
        
        # setting the title
        mns1.set_title("mns")
        col.set_title("color")
        def1.set_title("defiance mns")
        
        # showing the data
        mns1 = mns1.imshow(raster[0,:,:])
        col = col.imshow(raster[1,:,:], cmap="gray")
        def1 = def1.imshow(raster[2,:,:], cmap="gray")
        
        plt.show()
    
    else:
        # creating axes and figures
        fig, (mns1, col) = plt.subplots(1, 2, figsize=(14, 14)) # Create one plot with figure size 10 by 10
        
        # setting the title
        mns1.set_title("mns")
        col.set_title("color")
        
        # showing the data
        mns1 = mns1.imshow(raster[0,:,:])
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
        input = input[None,:,:,:].float().cuda()
    except:
        input = torch.from_numpy(input[None,:,:,:]).float().cuda()

    # loading the encoder
    model = trained_model.encoder
    
    # load altitude and reshape it
    alt = input[:,0,:,:][:,None,:,:]
    
    # load rad and reshape it
    rad = input[:,1,:,:][:,None,:,:]
    
    ## running the model
    # encoder alt
    a1 = model.sca2(model.ca1(alt))
    #level 2
    a3= model.sca4(model.ca3(a1))
    
    #encoder
    #level 1
    x1 = model.sc2(model.c1(rad))
    #level 2
    if args.data_fusion:
        x2= model.c3(x1 + a1)
        
        # extra layer
        x2_b = model.sc4(x2)
        
        #level 3
        x3 = model.c5(x2_b + a3)
    else:
        x2= model.c3(x1)
        
        # extra layer
        x2_b = model.sc4(x2)
        
        #level 3
        x3 = model.c5(x2_b)
    
    #decoder
    model = trained_model.decoder
    #level 2
    y4 = model.t1(model.c6(x3))
    y3 = model.c8(model.c7(y4))
    
    #level 1
    y2 = model.t2(y3)
    y1 = model.c10(model.c9(y2))
    
    #output        
    print(input.shape)
    
    # show input
    show(input[:,0,:,:].detach().cpu())
    show(input[:,1,:,:].detach().cpu())
    
    # show various embeddings in the model
    fig = plt.figure(figsize=(25, 10)) #adapted dimension
    ax = fig.add_subplot(3, 7, 2, aspect=1)
    ax.set(title='x1 : %d x %d x %d' %(x1.shape[1:]))
    view_embeddings(x1, ax)
    ax = fig.add_subplot(3, 7, 9, aspect=1)
    ax.set(title='x2 : %d x %d x %d' %(x2.shape[1:]))
    view_embeddings(x2, ax)
    ax = fig.add_subplot(3, 7, 10, aspect=1)
    ax.set(title='x2_b : %d x %d x %d' %(x2.shape[1:]))
    view_embeddings(x2_b, ax)
    ax = fig.add_subplot(3, 7, 17, aspect=1)
    ax.set(title='x3 : %d x %d x %d' %(x3.shape[1:]))
    
    if args.split:
        view_embeddings(x3[:,:8,:,:], ax)
        ax = fig.add_subplot(3, 7, 11, aspect=1)
        ax.set(title='y4 : %d x %d x %d' %(y4.shape[1:]))
        
    else:
        view_embeddings(x3, ax)
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
    
    return None


def change_detection(rast1, rast2, trained_model, args, gts = False, visualization=False, threshold=5):
  """
  param: two rasters of dims 1*2*128*128, our neural network model
  fun: outputs a change detection map based on two bi-temporal rasters
  """
  
  # ============rast1===========
  input = torch.from_numpy(rast1)
  input = input.float().cuda()
  
  # load rad and reshape it
  rad1 = input[:,1,:,:][:,None,:,:]
  alt1 = input[:,0,:,:][:,None,:,:]
  
  # loading the encoder
  trained_model = trained_model.encoder
  if args.split:
      code_rast1 = trained_model(input, data_fusion=args.data_fusion)[:,:8,:,:]
  else:
      code_rast1 = trained_model(input, data_fusion=args.data_fusion)
  
  # ============rast2===========
  input = torch.from_numpy(rast2)
  input = input.float().cuda()
  
  # load rad and reshape it
  rad2 = input[:,1,:,:][:,None,:,:]
  alt2 = input[:,0,:,:][:,None,:,:]

  #level 3
  if args.split:
      code_rast2 = trained_model(input, data_fusion=args.data_fusion)[:,:8,:,:]
  else:
      code_rast2 = trained_model(input, data_fusion=args.data_fusion)
      
  # ============cmap===========
  # difference matrix on the code
  CD_code = (code_rast1 - code_rast2) ** 2
  CD_code = torch.mean(CD_code, dim=1)
  CD_code = CD_code ** 0.5
  
  CD_code_cl = CD_code * (CD_code > threshold).float()
  
  # converting into numpy
  CD_code_cl = CD_code_cl.cpu().detach().numpy()
  
  # changing into a binary map
  non_zero_mat = np.nonzero(CD_code_cl)
  
  # creating the binary change map
  cmap_bin = CD_code_cl.copy()
  cmap_bin[non_zero_mat] = 1
  
  if visualization == True:
      # show various embeddings in the model
      fig = plt.figure(figsize=(25, 10)) #adapted dimension
      fig.suptitle("Change detection on two rasters threshold: {}".format(threshold))
      ax = fig.add_subplot(3, 7, 9, aspect=1)
      ax.set(title='Change map: float' )
      ax.imshow(CD_code.cpu().detach().numpy().squeeze(), cmap="hot")
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 2, aspect=1)
      ax.set(title='MNS 1' )
      ax.imshow(alt1.cpu().detach().numpy().squeeze())
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 16, aspect=1)
      ax.set(title='MNS 2' )
      ax.imshow(alt2.cpu().detach().numpy().squeeze())
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 1, aspect=1)
      ax.set(title='Raster 1' )
      ax.imshow(rad1.cpu().numpy().squeeze(), cmap="gray")
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 15, aspect=1)
      ax.set(title='Raster 2' )
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
          gt_map[data_index] = cmap_gt
          
          # loading the gt values
          gt_map[data_index] = cmap_gt
          
          ## calcuating the roc
          # getting the difference raster to same dimensions
          CD_code = CD_code.detach().cpu().numpy()
          pred_map = regrid(CD_code.squeeze().reshape(CD_code.shape[1:]), 128, 128, "nearest")
          
          # loading the predicted values
          pred_change = pred_map[data_index]
          
          ax = fig.add_subplot(3, 7, 4, aspect=1)
          ax.set(title='GT raster 1' )
          ax.imshow(gts[0], cmap=cmap, norm=norm)
          plt.axis('off')
          
          ax = fig.add_subplot(3, 7, 18, aspect=1)
          ax.set(title='GT raster 2' )
          ax.imshow(gts[1], cmap=cmap, norm=norm)
          plt.axis('off')
          
          ax = fig.add_subplot(3, 7, 11, aspect=1)
          ax.set(title='GT cmap, Nodata is -1' )
          ax.imshow(gt_map, cmap=colors_cmap, norm=norm_cmap)
          plt.axis('off')
          
          
          
          
          # we make the roc analysis if there is relevant GT data
          try:
              
              # removing no data values
              diff_mns = diff_mns.detach().cpu().numpy().squeeze()[data_index]
              diff_radio = diff_radio.detach().cpu().numpy().squeeze()[data_index]
              
              ## getting roc for the baseline
              fpr_alt, tpr_alt, thresholds = metrics.roc_curve(cmap_gt, diff_mns)
              fpr_rad, tpr_rad, thresholds = metrics.roc_curve(cmap_gt, diff_radio)
              
              # getting roc values
              fpr, tpr, thresholds = metrics.roc_curve(cmap_gt, pred_change)
              auc = metrics.roc_auc_score(cmap_gt, pred_change)
              
              
              ax = fig.add_subplot(3, 7, 8, aspect=1)
              ax.set(title='ROC curve, AUC: %1.2f' % (auc))
              ax.plot(fpr, tpr, linestyle='--')
              ax.plot(fpr_alt, tpr_alt, linestyle=':')
              ax.plot(fpr_rad, tpr_rad, linestyle='-')
             
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
                try:
                    # cropping the raster
                    out_img, out_transform = mask(dataset=rast, all_touched=True,
                                                  shapes=our_box, crop=True)
                    
                    # storing the raster in our list
                    # removing rasters with too much zeros
                    values = out_img.flatten()
                    nb_zeroes = np.count_nonzero(values == 0)
                    
                    if nb_zeroes > len(values)/5 :
                        None
                    else:
                        resh_rast = regrid(out_img.reshape(out_img.shape[1:]), 128, 128)
                        rasters_box.append(resh_rast)
                
                except:
                    None
        
        else: 
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
    
    
    code = code.cpu().detach().numpy().squeeze().reshape(16,1024)

    data_pca = PCA(n_components=3)
    data_pca.fit(code)
    
    data_pc = data_pca.components_.reshape(3,32,32)
    
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
    
    result = torch.from_numpy(raster).cuda().float()
    
    return result


def prepare_nmi(list_rasters, discrete=False):
    
    # llist to store the rasters
    reshap_rasts = []
    
    # reshaping and loading in the list
    for rast in list_rasters:
        
        # reshaping
        rast_resh =  regrid(rast.reshape(rast.shape), 32, 32, "nearest")
        
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











