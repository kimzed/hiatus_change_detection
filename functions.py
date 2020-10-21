# Project hiatus
# functions
# 21/10/2020
# Cédric BARON

# importing libraries
import json
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.optim as optim
import math
from sklearn.decomposition import PCA

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
        
        
def train_val_dataset(dataset, val_split=0.25):
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
    
    return datasets


class SegNet(nn.Module):
  """
  EncoderDecoder network for semantic segmentation
  """
  
  def __init__(self, n_channels, encoder_conv_width, decoder_conv_width, cuda = False):
    """
    initialization function
    n_channels, int, number of input channel
    encoder_conv_width, int list, size of the feature maps depth for the encoder after each conv
    decoder_conv_width, int list, size of the feature maps depth for the decoder after each conv
    n_class = int,  the number of classes
    """
    super(SegNet, self).__init__() #necessary for all classes extending the module class
    
    self = self.float()
    
    # here the index 1 is for after the second CN which is used by maxpool (2 for one block)
    assert((encoder_conv_width[3] == encoder_conv_width[5]) \
     and (encoder_conv_width[1] == decoder_conv_width[1]))
    
    self.maxpool=nn.MaxPool2d(2,2,return_indices=True) #maxpooling layer
    
    #encoder
    #usage:
    #nn.Conv2d(depth_of_input, depth_of_output,size_of_kernel (3),padding=1, padding_mode='reflection')
    #nn.BatchNorm2d(depth_of_layer)
    # n_channels is the number of channels from the input
    self.c1 = nn.Sequential(nn.Conv2d(n_channels, encoder_conv_width[0],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[0]),nn.ReLU(True))
    self.c2 = nn.Sequential(nn.Conv2d(encoder_conv_width[0],encoder_conv_width[1],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[1]),nn.ReLU(True))
    self.c3 = nn.Sequential(nn.Conv2d(encoder_conv_width[1],encoder_conv_width[2],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[2]),nn.ReLU(True))
    self.c4 = nn.Sequential(nn.Conv2d(encoder_conv_width[2],encoder_conv_width[3],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[3]),nn.ReLU(True))
    self.c5 = nn.Sequential(nn.Conv2d(encoder_conv_width[3],encoder_conv_width[4],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[4]),nn.ReLU(True))
    self.c6 = nn.Sequential(nn.Conv2d(encoder_conv_width[4],encoder_conv_width[5],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[5]),nn.ReLU(True))
    #decoder
    # the extra width is added because of concatenation ?
    self.t1 = nn.Sequential(nn.ConvTranspose2d(encoder_conv_width[5], encoder_conv_width[5], 2, 2), nn.BatchNorm2d(encoder_conv_width[5]),nn.ReLU(True))
    self.c7 = nn.Sequential(nn.Conv2d(encoder_conv_width[5],decoder_conv_width[0],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[0]),nn.ReLU(True))
    self.c8 = nn.Sequential(nn.Conv2d(decoder_conv_width[0],decoder_conv_width[1],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[1]),nn.ReLU(True))
    self.t2 = nn.Sequential(nn.ConvTranspose2d(decoder_conv_width[1], decoder_conv_width[1], 2, 2), nn.BatchNorm2d(decoder_conv_width[1]),nn.ReLU(True))
    self.c9 = nn.Sequential(nn.Conv2d(decoder_conv_width[1],decoder_conv_width[2],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[2]),nn.ReLU(True))
    self.c10 = nn.Sequential(nn.Conv2d(decoder_conv_width[2],decoder_conv_width[3],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[3]),nn.Dropout(0.7), nn.ReLU(True)) 
    
    # network for the altitude
    self.a1 = nn.Sequential(nn.Conv2d(n_channels, encoder_conv_width[0],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[0]),nn.ReLU(True))
    self.a2 = nn.Sequential(nn.Conv2d(encoder_conv_width[0],encoder_conv_width[1],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[1]),nn.ReLU(True))
    self.a3 = nn.Sequential(nn.Conv2d(encoder_conv_width[1],encoder_conv_width[2],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[2]),nn.ReLU(True))
    self.a4 = nn.Sequential(nn.Conv2d(encoder_conv_width[2],encoder_conv_width[3],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[3]),nn.ReLU(True))
    
    #final  layer
    self.final=nn.Conv2d(decoder_conv_width[3],2,3,padding=1, padding_mode='reflect')

    #weight initialization

    self.c1[0].apply(self.init_weights)
    self.c2[0].apply(self.init_weights)
    self.c3[0].apply(self.init_weights)
    self.c4[0].apply(self.init_weights)
    self.c5[0].apply(self.init_weights)
    self.c6[0].apply(self.init_weights)
    self.t1[0].apply(self.init_weights)
    self.c7[0].apply(self.init_weights)
    self.c8[0].apply(self.init_weights)
    self.t2[0].apply(self.init_weights)
    self.c9[0].apply(self.init_weights)
    self.c10[0].apply(self.init_weights)
    
    # running the model on gpu
    self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
  def forward(self,input):
    """
    the function called to run inference
    after the model is created as an object 
    we call the method with the input as an argument
    """
    #encoder altitude
    #level 1
    a1 = self.a2(self.a1(input[:,0,:,:].view(input.shape[0], 1, input.shape[2], input.shape[3])))
    a2, indices_a_b = self.maxpool(a1)
    #level 2
    a3= self.a4(self.a3(a2))
    a4, indices_b_c = self.maxpool(a3)
    
    #encoder visual
    #level 1
    x1 = self.c2(self.c1(input[:,1,:,:].view(input.shape[0], 1, input.shape[2], input.shape[3])))
    x2, indices_a_b = self.maxpool(x1)
    #level 2
    x3= self.c4(self.c3(x2 + a2))
    x4, indices_b_c = self.maxpool(x3)
    #level 3
    x5 = self.c6(self.c5(x4 + a4))
    
    #decoder
    #level 2
    y4 = self.t1(x5)
    y3 = self.c8(self.c7(y4))
    
    #level 1       
    y2 = self.t2(y3)
    y1 = self.c10(self.c9(y2))
    out = self.final(y1)

    return out

def train(model, optimizer, args, datasets):
  """
  train for one epoch
  args are some parameters of our model, e.g. batch size or n_class, etc.
  """
  
  #switch the model in training mode
  model.train() 
  
  #the loader function will take care of the batching
  # train_set was defined prior
  loader = torch.utils.data.DataLoader(datasets["train"], \
         batch_size=args.batch_size, shuffle=True, drop_last=True)
  
  # loss on the whole dataset
  loss_data = 0.0
  loss_data_alt = 0.0
  loss_data_rad = 0.0
  
  # loops over the batches
  for index, tiles in enumerate(loader):
    
    # loading on the gpu
    tiles = tiles.cuda().float()
    
    optimizer.zero_grad() #put gradient to zero
    
    pred = model(tiles) #compute the prediction
    
    # boolean matrixes to remove effect of no data
    bool_matr_alt = tiles[:,None,0,:,:] != 0
    bool_matr_rad = tiles[:,None,1,:,:] != 0
    
    # filtering the data
    pred_alt = pred[:,None,0,:,:][bool_matr_alt]
    tiles_alt = tiles[:,None,0,:,:][bool_matr_alt]
    pred_rad = pred[:,None,1,:,:][bool_matr_rad]
    tiles_rad = tiles[:,None,1,:,:][bool_matr_rad]
    
    ## sum of squares
    loss_fun = nn.MSELoss()
    loss_alt = loss_fun(pred_alt, tiles_alt)
    loss_rad = loss_fun(pred_rad, tiles_rad)
    
    # total loss
    loss = loss_alt*0.2 + loss_rad*0.8
    
    # cross entropy
    #loss = nn.functional.binary_cross_entropy(pred[tiles != 0], tiles[tiles != 0])
    
    loss_data += loss.item()
    loss_data_alt += loss_alt.item()
    loss_data_rad += loss_rad.item()
    
    loss.backward() #compute gradients

    #for p in model.parameters(): #we clip the gradient at norm 1
    #  p.grad.data.clamp_(-1, 1) #this helps learning faster
    
    # stochastic gradient descent
    optimizer.step() #one SGD step
    
  return loss_data, len(loader), loss_data_alt, loss_data_rad


def train_full(args, datasets):
  """
  The full training loop
  """
  
  #initialize the model
  model = SegNet(args.n_channel, args.conv_width, args.dconv_width)

  print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
  
  #define the optimizer
  #adam optimizer is always a good guess for classification
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  TRAINCOLOR = '\033[100m'
  NORMALCOLOR = '\033[0m'
  
  # storing losses to display them eventually
  losses = {"tot":[], "mns":[], "alt":[]}
  
  for i_epoch in range(args.n_epoch):
      
    #train one epoch
    loss_train, nb_batches, loss_alt, loss_rad = train(model, optimizer, args, datasets)
    
    loss = math.sqrt((loss_train / nb_batches) / 64)
    
    # storing loss for later plotting
    losses["tot"].append(loss)
    losses["mns"].append(math.sqrt((loss_alt/ nb_batches) / 64))
    losses["alt"].append(math.sqrt((loss_rad/ nb_batches) / 64))
    
    print(TRAINCOLOR)
    print('Epoch %3d -> Train Loss: %1.4f' % (i_epoch, loss) + NORMALCOLOR)
    print("loss mns is %1.4f" % (loss_alt))
    print("loss rad is %1.4f" % (loss_rad))
    
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


  return model


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
  

def view_u(train, tile_index = None, trained_model):
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
  
  #decoder
  #level 2
  y4 = trained_model.t1(CD_code)
  y3 = trained_model.c8(trained_model.c7(y4))
  
  #level 1       
  y2 = trained_model.t2(y3)
  y1 = trained_model.c10(trained_model.c9(y2))
  #output         
  out = trained_model.final(y1)
  
  return out.detach().cpu(), CD_code, code_rast1, code_rast2