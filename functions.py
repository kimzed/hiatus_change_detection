# Project hiatus
# functions
# 21/10/2020
# Cédric BARON

# importing libraries
import json
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from rasterio.plot import show

# this is used for the visualize function
from mpl_toolkits.mplot3d import Axes3D

def getFeatures(gdf):
    """
    param: a geopanda dataframe
    Function to parse features from GeoDataFrame in such a manner that rasterio wants them
    """
    
    return [json.loads(gdf.to_json())['features'][0]['geometry']]



def regrid(data, out_x, out_y):
    """
    param: numpy array, number of columns, number of rows
    fun: function to interpolate a raster
    
    """
    
    m = max(data.shape[0], data.shape[1])
    y = np.linspace(0, 1.0/m, data.shape[0])
    x = np.linspace(0, 1.0/m, data.shape[1])
    interpolating_function = RegularGridInterpolator((y, x), data)
    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, out_y), np.linspace(0, 1.0/m, out_x))
    
    # reprojects the data
    return interpolating_function((xv, yv))



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
    input = input.view(1, input.shape[0], input.shape[1], input.shape[2]).float().cuda()

    ## running the model
    # encoder alt
    a1 = trained_model.a2(trained_model.a1(input[:,0,:,:].view(input.shape[0], 1,
                                           input.shape[2], input.shape[3])))
    a2, indices_a_b = trained_model.maxpool(a1)
    #level 2
    a3= trained_model.a4(trained_model.a3(a2))
    a4, indices_b_c = trained_model.maxpool(a3)
    
    #encoder
    #level 1
    x1 = trained_model.c2(trained_model.c1(input[:,1,:,:].view(input.shape[0], 1,
                                           input.shape[2], input.shape[3])))
    x2, indices_a_b = trained_model.maxpool(x1)
    #level 2
    x3= trained_model.c4(trained_model.c3(x2 + a2))
    x4, indices_b_c = trained_model.maxpool(x3)
    #level 3
    x5 = trained_model.c6(trained_model.c5(x4 + a4))
    #decoder
    #level 2
    y4 = trained_model.t1(x5)
    y3 = trained_model.c8(trained_model.c7(y4))
    
    #level 1       
    y2 = trained_model.t2(y3)
    y1 = trained_model.c10(trained_model.c9(y2))
    #output         
    out = trained_model.final(y1)
    print(out.shape)
    
    # show input
    show(input[:,0,:,:].detach().cpu())
    show(input[:,1,:,:].detach().cpu())
    
    # show output
    show(out[:,0,:,:].detach().cpu())
    show(out[:,1,:,:].detach().cpu())
    
    # show various embeddings in the model
    fig = plt.figure(figsize=(25, 10)) #adapted dimension
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
    view_embeddings(x4, ax)
    ax = fig.add_subplot(3, 7, 18, aspect=1)
    ax.set(title='x5 : %d x %d x %d' %(x5.shape[1:]))
    view_embeddings(x5, ax)
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



def change_detection(rast1, rast2, trained_model, threshold):
  """
  param: two rasters of dims 1*2*128*128, our neural network model
  fun: outputs a change detection map based on two bi-temporal rasters
  """
  
  ## encoding rast1
  input = torch.from_numpy(rast1)
  input = input.float().cuda()
  
  # encoder alt
  a1 = trained_model.a2(trained_model.a1(input[:,0,:,:].view(input.shape[0], 1,
                                         input.shape[2], input.shape[3])))
  a2, indices_a_b = trained_model.maxpool(a1)
  #level 2
  a3= trained_model.a4(trained_model.a3(a2))
  a4, indices_b_c = trained_model.maxpool(a3)
  #encoder
  #level 1
  x1 = trained_model.c2(trained_model.c1(input[:,1,:,:].view(input.shape[0], 1,
                                         input.shape[2], input.shape[3])))
  x2, indices_a_b = trained_model.maxpool(x1)
  #level 2
  x3= trained_model.c4(trained_model.c3(x2 + a2))
  x4, indices_b_c = trained_model.maxpool(x3)
  ## code
  code_rast1 = trained_model.c6(trained_model.c5(x4 + a4))
  
  ## encoding rast2
  input = torch.from_numpy(rast2)
  input = input.float().cuda()
  # encoder alt
  a1 = trained_model.a2(trained_model.a1(input[:,0,:,:].view(input.shape[0], 1,
                                         input.shape[2], input.shape[3])))
  a2, indices_a_b = trained_model.maxpool(a1)
  #level 2
  a3= trained_model.a4(trained_model.a3(a2))
  a4, indices_b_c = trained_model.maxpool(a3)
  #encoder
  #level 1
  x1_2 = trained_model.c2(trained_model.c1(input[:,1,:,:].view(input.shape[0], 1,
                                         input.shape[2], input.shape[3])))
  
  x2, indices_a_b = trained_model.maxpool(x1_2)
  #level 2
  x3_2= trained_model.c4(trained_model.c3(x2 + a2))
  x4, indices_b_c = trained_model.maxpool(x3_2)
  ## code
  code_rast2 = trained_model.c6(trained_model.c5(x4 + a4))
  
  ## difference matrixes on several levels
  # difference matrix on the code
  CD_code = torch.abs(code_rast1 - code_rast2)
  CD_code = code_rast2 * (CD_code > threshold).float()
  
  return CD_code, code_rast1, code_rast2