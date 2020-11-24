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
from sklearn.metrics import confusion_matrix
import random

# this is used for the visualize function
from mpl_toolkits.mplot3d import Axes3D

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



def visualize(raster, third_dim=True):
    """
    param: a raster 2*128*128, with mns and radiometry
    fun: visualize a given raster in two dimensions and in 3d for altitude
    """
    
    # creating axes and figures
    fig, (mns1, col) = plt.subplots(1, 2, figsize=(14, 14)) # Create one plot with figure size 10 by 10
    
    # setting the title
    mns1.set_title("mns")
    col.set_title("color")
    
    # showing the data
    mns1 = mns1.imshow(raster[0,:,:])
    col = col.imshow(raster[1,:,:])
    
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
    

def view_embeddings(fmap, ax = None):
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
    
    plt.axis('off')
    

def view_u(train, trained_model, tile_index = None):
    """
    param: datasets, index of the raster, AE model
    fun: runs the model on the data and visualize various embeddings inside
         the model
    """
    
    # loading the data and reshaping it for prediction
    input = train[tile_index]
    
    # converting to adequate format
    try:
        input = input.view(1, input.shape[0], input.shape[1], input.shape[2]).float().cuda()
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
    x2= model.c3(x1 + a1)
    
    # extra layer
    x2_b = model.sc4(x2)
    
    #level 3
    x3 = model.c5(x2_b + a3)
    
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


def change_detection(rast1, rast2, trained_model, threshold, gts = False, visualization=False):
  """
  param: two rasters of dims 1*2*128*128, our neural network model
  fun: outputs a change detection map based on two bi-temporal rasters
  """
  
  # ============rast1===========
  input = torch.from_numpy(rast1)
  input = input.float().cuda()
  
  # load altitude and reshape it
  alt = input[:,0,:,:][:,None,:,:]
    
  # load rad and reshape it
  rad1 = input[:,1,:,:][:,None,:,:]
  
  # loading the encoder
  trained_model = trained_model.encoder
  
  # encoder alt
  a1 = trained_model.sca2(trained_model.ca1(alt))
  #level 2
  a3= trained_model.sca4(trained_model.ca3(a1))

  #encoder
  #level 1
  x1 = trained_model.sc2(trained_model.c1(rad1))
  #level 2
  x2= trained_model.sc4(trained_model.c3(x1 + a1))
  #level 3
  code_rast1 = trained_model.c5(x2 + a3)
  
  # ============rast2===========
  input = torch.from_numpy(rast2)
  input = input.float().cuda()
  
  # load altitude and reshape it
  alt = input[:,0,:,:][:,None,:,:]
    
  # load rad and reshape it
  rad2 = input[:,1,:,:][:,None,:,:]
  
  # encoder alt
  a1 = trained_model.sca2(trained_model.ca1(alt))
  #level 2
  a3= trained_model.sca4(trained_model.ca3(a1))

  #encoder
  #level 1
  x1 = trained_model.sc2(trained_model.c1(rad2))
  #level 2
  x2= trained_model.sc4(trained_model.c3(x1 + a1))
  #level 3
  code_rast2 = trained_model.c5(x2 + a3)
  
  # ============cmap===========
  # difference matrix on the code
  CD_code = (code_rast1 - code_rast2) ** 2
  CD_code = torch.sum(CD_code, dim=1)
  CD_code = CD_code ** 0.5
  
  CD_code_cl = CD_code * (CD_code > threshold).float()
  
  # converting into numpy
  CD_code_cl = CD_code_cl.cpu().detach().numpy()
  
  # changing into a binary map
  non_zero_alt = np.nonzero(CD_code_cl)
  
  # creating the binary change map
  cmap_bin = CD_code_cl.copy()
  cmap_bin[non_zero_alt] = 1
  
  if visualization:
      # show various embeddings in the model
      fig = plt.figure(figsize=(25, 10)) #adapted dimension
      fig.suptitle("Change detection on two rasters")
      ax = fig.add_subplot(3, 7, 9, aspect=1)
      ax.set(title='Change map: float' )
      ax.imshow(CD_code.cpu().detach().numpy().squeeze())
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 1, aspect=1)
      ax.set(title='Raster 1' )
      ax.imshow(rad1.cpu().numpy().squeeze())
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 15, aspect=1)
      ax.set(title='Raster 2' )
      ax.imshow(rad2.cpu().numpy().squeeze())
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 10, aspect=1)
      ax.set(title='Change map binary, min value= ' + str(cmap_bin.min()) )
      ax.imshow(cmap_bin.squeeze())
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 2, aspect=1)
      ax.set(title='Code raster 1' )
      view_embeddings(code_rast1, ax)
      plt.axis('off')
      
      ax = fig.add_subplot(3, 7, 16, aspect=1)
      ax.set(title='Code raster 2' )
      view_embeddings(code_rast2, ax)
      plt.axis('off')
      
      
      
      if gts:
          ax = fig.add_subplot(3, 7, 3, aspect=1)
          ax.set(title='GT raster 1' )
          ax.imshow(gts[0])
          plt.axis('off')
          
          ax = fig.add_subplot(3, 7, 17, aspect=1)
          ax.set(title='GT raster 2' )
          ax.imshow(gts[1])
          plt.axis('off')
      
  
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
    

class ConfusionMatrixBinary:
  def __init__(self, n_class, class_names):
    self.CM = np.zeros((n_class, n_class))
    self.n_class = n_class
    self.class_names = class_names
  
  def clear(self):
    self.CM = np.zeros((self.n_class, self.n_class))
    
  def add_batch(self, gt, pred):
    self.CM +=  confusion_matrix(gt, pred, labels = list(range(self.n_class)))
    
  def overall_accuracy(self):#percentage of correct classification
    return np.trace(self.CM) / np.sum(self.CM)

  def class_IoU(self, show = 1):
    ious = np.full(self.n_class, 0.)
    for i_class in range(self.n_class):
      error_matrix = [i for i in range(self.n_class) if i != i_class]
      ious[i_class] = self.CM[i_class, i_class] / (np.sum(self.CM[i_class, error_matrix]) + np.sum(self.CM[error_matrix, i_class]) + self.CM[i_class, i_class])
    if show:
      print('  |  '.join('{} : {:3.2f}%'.format(name, 100*iou) for name, iou in zip(self.class_names,ious)))
    #do not count classes that are not present in the dataset in the mean IoU
    return 100*np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()


def accuracy_model(list_rast_gt, model, threshold):
    
    # loading the list of years
    years = list(list_rast_gt.keys())
        
    # numb test
    nb_test = 100
    
    # loading the confusion matrix 
    m = ConfusionMatrixBinary(2, ["change", "no change"])
    
    
    # getting accuracy score on random combinations
    for i in range(nb_test):
        
        # getting two years randomly
        year1 = years[random.randint(0, len(years)-1)]
        year2 = years[random.randint(0, len(years)-1)]
        
        # getting a random integer
        ind = random.randint(0, len(list_rast_gt[year1])-1)
        
        # loading the two rasters
        rast1 = list_rast_gt[year1][ind]
        rast2 = list_rast_gt[year2][ind]
        
        # loading the rasters
        gt1 = rast1[0,:,:]
        rast1 = rast1[1:,:,:][None,:,:,:]
        
        # loading the rasters
        gt2 = rast2[0,:,:]
        rast2 = rast2[1:,:,:][None,:,:,:]
        
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
        
        # computing change raster
        cmap, dccode, code1, code2 = change_detection(rast1, rast2, model,
                                                      threshold, (gt1,gt2), visualization=False)
        
        # reshaping to original dimensions
        cmap = regrid(cmap.reshape(cmap.shape[1:]), 128, 128, "nearest")
        cmap_cl = np.rint(cmap)
        
        # removing no data
        cmap_cl = cmap_cl[data_index]
        
        # putting into the confusion matrix
        m.add_batch(cmap_gt, cmap_cl)
    
    return m
