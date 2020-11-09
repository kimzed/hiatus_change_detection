# Project hiatus
# model
# 02/11/2020
# Cédric BARON

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import math
import torch.nn.functional as F
import torchnet as tnt
import numpy as np

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
    
    # softplus for the defiance
    self.sfplus=nn.Softplus()
    
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
    self.c10 = nn.Sequential(nn.Conv2d(decoder_conv_width[2],decoder_conv_width[3],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[3]), nn.ReLU(True)) 
    
    # network for the altitude
    self.a1 = nn.Sequential(nn.Conv2d(n_channels, encoder_conv_width[0],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[0]),nn.ReLU(True))
    self.a2 = nn.Sequential(nn.Conv2d(encoder_conv_width[0],encoder_conv_width[1],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[1]),nn.ReLU(True))
    self.a3 = nn.Sequential(nn.Conv2d(encoder_conv_width[1],encoder_conv_width[2],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[2]),nn.ReLU(True))
    self.a4 = nn.Sequential(nn.Conv2d(encoder_conv_width[2],encoder_conv_width[3],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[3]),nn.ReLU(True))
    
    #final  layer
    self.final = nn.Conv2d(decoder_conv_width[3],4,3,padding=1, padding_mode='reflect')

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
    self.final.apply(self.init_weights)
    
    # for the DEM part
    self.a1[0].apply(self.init_weights)
    self.a2[0].apply(self.init_weights)
    self.a3[0].apply(self.init_weights)
    self.a4[0].apply(self.init_weights)
    
    # running the model on gpu
    self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
    #nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity="leaky_relu")
    nn.init.normal_(layer.weight, mean=5, std=3)
    
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
    y = self.final(y1)
    out = torch.cat((y[:,0:2,:,:], self.sfplus(y[:,None,2,:,:]), self.sfplus(y[:,None,3,:,:])), 1)
    
    return out, x5


class Discriminator(nn.Module):
  """
  #Discriminator network for year detection
  """
  
  def __init__(self, nb_years=5):
    """
    #initialization function
    #n_channels, int, number of input channel
    #encoder_conv_width, int list, size of the feature maps depth for the encoder after each conv
    #decoder_conv_width, int list, size of the feature maps depth for the decoder after each conv
    #n_class = int,  the number of classes
    """
    super(Discriminator, self).__init__() #necessary for all classes extending the module class
    
    self = self.float()
    
    self.maxpool=nn.MaxPool2d(2,2)
    
    # here the convolutions don't change the width and height, only the number of channels
    self.fc1 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.fc2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.fc3 = nn.Sequential(nn.Conv2d(16, 4, 3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(4),nn.ReLU(True))
    self.fc4 = nn.Sequential(nn.Conv2d(4, 5, 1, padding=0, padding_mode='reflect')) #conv (1x1)

    # initiating weights
    self.fc1[0].apply(self.init_weights)
    self.fc2[0].apply(self.init_weights)
    self.fc3[0].apply(self.init_weights)
    self.fc4[0].apply(self.init_weights)
    
    self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
      nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
    
  def forward(self, x):
    """
    #here x is the input
    """
    
    out = self.fc4(self.fc3(self.fc2(self.fc1(x))))

    return out

def train(model, discr, optimizer, optimizer_D, args, datasets):
  """
  train for one epoch
  args are some parameters of our model, e.g. batch size or n_class, etc.
  """
  
  
  #switch the model in training mode
  model.train() 
  
  #the loader function will take care of the batching
  # train_set was defined prior
  loader = torch.utils.data.DataLoader(datasets, \
         batch_size=args.batch_size, shuffle=True, drop_last=True)
  
  # loss on the whole dataset
  loss_data = tnt.meter.AverageValueMeter()
  loss_data_alt = tnt.meter.AverageValueMeter()
  loss_data_rad = tnt.meter.AverageValueMeter()
  loss_disc_val = tnt.meter.AverageValueMeter()
  
  # loops over the batches
  for index, (tiles, labels) in enumerate(loader):
    
    # loading on the gpu
    tiles = tiles.cuda().float()
    labels = labels.cuda().long()
    
    # ============forward===========
    # compute the prediction
    pred, code = model(tiles) 
    
    # boolean matrixes to remove effect of no data
    bool_matr_alt = tiles[:,None,0,:,:] != 0
    bool_matr_rad = tiles[:,None,1,:,:] != 0
    
    # filtering the data
    pred_alt = pred[:,None,0,:,:][bool_matr_alt]
    tiles_alt = tiles[:,None,0,:,:][bool_matr_alt]
    pred_rad = pred[:,None,1,:,:][bool_matr_rad]
    tiles_rad = tiles[:,None,1,:,:][bool_matr_rad]
    
    # loading defiance matrix
    d_mat_alt = pred[:,None,2,:,:][bool_matr_alt]
    d_mat_rad = pred[:,None,3,:,:][bool_matr_rad]
    
    ## sum of squares
    #loss_fun = nn.MSELoss()
    #loss_alt = loss_fun(tiles_alt, pred_alt)
    #loss_rad = loss_fun(tiles_rad, pred_rad)
    loss_alt = torch.mean(torch.abs(tiles_alt - pred_alt) / (2*d_mat_alt**2) + 2*torch.log(d_mat_alt))
    loss_rad = torch.mean(torch.abs(tiles_rad - pred_rad) / (2*d_mat_rad**2) + 2*torch.log(d_mat_rad))
    
    # reshaping the labels 
    list_labels = [labels for i in range(code.shape[-1])]
    labels = torch.stack(list_labels, dim=-1)
    list_labels = [labels for i in range(code.shape[-1])]
    labels = torch.stack(list_labels, dim=-1)
    _, labels = labels.max(dim=1)
    
    for i in range(2):
        ## now the disciminant part
        pred_year = discr(code.detach())
        criterion =  nn.CrossEntropyLoss(reduction="none")
        
        # applying loss function
        loss_disc = criterion(pred_year, labels)
        loss_disc = loss_disc.mean()
        optimizer_D.zero_grad()
        loss_disc.backward()
        optimizer_D.step()
        loss_disc_val.add(loss_disc.item())
    
    pred_year = discr(model(tiles)[1])
    loss_disc = criterion(pred_year, labels)
    loss_disc = loss_disc.mean()
    
    # total loss
    loss = loss_alt + loss_rad - 0.8 * loss_disc
    loss_data.add(loss.item())
    
    # ============backward===========
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step() #one SGD step
    
    # storing the loss values
    loss_data_alt.add(loss_alt.cpu().detach())
    loss_data_rad.add(loss_rad.cpu().detach())


    #for p in model.parameters(): #we clip the gradient at norm 1
    #  p.grad.data.clamp_(-1, 1) #this helps learning faster
    
  return loss_data.value()[0], len(loader), loss_data_alt.value()[0], loss_data_rad.value()[0], loss_disc_val.value()[0]


def train_full(args, datasets, writer):
  """
  The full training loop
  """
  
  #initialize the models
  model = SegNet(args.n_channel, args.conv_width, args.dconv_width)
  discr = Discriminator()
  #discr = Discriminator()

  print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
  
  #define the optimizer
  #adam optimizer is always a good guess for classification
  optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
  optimizer_D = optim.Adam(discr.parameters(), lr=args.lr)
  TRAINCOLOR = '\033[100m'
  NORMALCOLOR = '\033[0m'
  
  # storing losses to display them eventually
  losses = {"tot":[], "mns":[], "alt":[]}
  
  for i_epoch in range(args.n_epoch):
      
    #train one epoch
    loss_train, nb_batches, loss_alt, loss_rad, loss_disc = train(model, discr, optimizer, optimizer_D, args, datasets)
    
    loss = loss_train
    loss_alt = loss_alt
    loss_rad = loss_rad
    #loss_disc = loss_disc / (nb_batches * args.batch_size)
    
    # storing loss for later plotting
    losses["tot"].append(loss)
    losses["mns"].append(loss_alt)
    losses["alt"].append(loss_rad)
    #losses["discr"].append(loss_disc)
    
    print(TRAINCOLOR)
    print('Epoch %3d -> Train Loss: %1.4f' % (i_epoch, loss) + NORMALCOLOR)
    print("loss mns is %1.4f" % (loss_alt))
    print("loss rad is %1.4f" % (loss_rad))
    print("loss discr is %1.4f" % (loss_disc))
    
    
    # ...log the running loss
    writer.add_scalar('training loss',
                    loss,
                    i_epoch)
    
    writer.add_scalar('altitude loss',
                    loss_alt,
                    i_epoch)
    
    writer.add_scalar('radiometric loss',
                    loss_rad,
                    i_epoch)
    
    writer.add_scalar('discriminator loss',
                    loss_disc,
                    i_epoch)
    
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
  
  #plt.title('loss discr per number of epochs')
  #plt.xlabel('epoch')
  #plt.ylabel('loss')
  #plt.plot(range(len(losses["discr"])), losses["discr"])
  #plt.show()

  return model, discr