# Project hiatus
# model
# 02/11/2020
# Cédric BARON

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
import numpy as np
import os

os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

# importing our functions
import loss as loss_fun

class AutoEncoder(nn.Module):
  """
  EncoderDecoder network for semantic segmentation
  """
  
  def __init__(self, encoder, decoder, opti_E, opti_D):
      
      super(AutoEncoder, self).__init__()
      
      # saving the two models in the object
      self.encoder = encoder
      self.decoder = decoder
      
      # saving the optimizers in the object
      self.opti_E = opti_E
      self.opti_D = opti_D
      
      
  def predict(self, input):
      
      # compute code and output
      code = self.encoder(input)
      out = self.decoder(code)
      
      return out
  
    
  def code(self, input):
    
      # computes the code
      code = self.encoder(input)
      
      return code
  
    
class Encoder(nn.Module):
  """
  EncoderDecoder network for semantic segmentation
  """
  
  def __init__(self, n_channels, encoder_conv_width, cuda = False):
    """
    initialization function
    n_channels, int, number of input channel
    encoder_conv_width, int list, size of the feature maps depth for the encoder after each conv
    decoder_conv_width, int list, size of the feature maps depth for the decoder after each conv
    n_class = int,  the number of classes
    """
    super(Encoder, self).__init__() #necessary for all classes extending the module class
    
    self = self.float()
    
    # softplus for the defiance
    self.sfplus=nn.Softplus()
    
    #encoder
    #usage:
    #nn.Conv2d(depth_of_input, depth_of_output,size_of_kernel (3),padding=1, padding_mode='reflection')
    #nn.BatchNorm2d(depth_of_layer)
    # n_channels is the number of channels from the input
    self.c1 = nn.Sequential(nn.Conv2d(n_channels, encoder_conv_width[0],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[0]),nn.LeakyReLU(True))
    self.c2 = nn.Sequential(nn.Conv2d(encoder_conv_width[0],encoder_conv_width[1],4,padding=1, stride=2, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[1]),nn.LeakyReLU(True))
    self.c3 = nn.Sequential(nn.Conv2d(encoder_conv_width[1],encoder_conv_width[2],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[2]),nn.LeakyReLU(True))
    self.c4 = nn.Sequential(nn.Conv2d(encoder_conv_width[2],encoder_conv_width[3],4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[3]),nn.LeakyReLU(True))
    self.c5 = nn.Sequential(nn.Conv2d(encoder_conv_width[3],encoder_conv_width[4],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[4]),nn.LeakyReLU(True))
    
    # network for the altitude
    self.a1 = nn.Sequential(nn.Conv2d(n_channels, encoder_conv_width[0],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[0]),nn.LeakyReLU(True))
    self.a2 = nn.Sequential(nn.Conv2d(encoder_conv_width[0],encoder_conv_width[1],4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[1]),nn.LeakyReLU(True))
    self.a3 = nn.Sequential(nn.Conv2d(encoder_conv_width[1],encoder_conv_width[2],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[2]),nn.LeakyReLU(True))
    self.a4 = nn.Sequential(nn.Conv2d(encoder_conv_width[2],encoder_conv_width[3],4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[3]),nn.LeakyReLU(True))

    #weight initialization
    self.c1[0].apply(self.init_weights)
    self.c2[0].apply(self.init_weights)
    self.c3[0].apply(self.init_weights)
    self.c4[0].apply(self.init_weights)
    self.c5[0].apply(self.init_weights)
    
    # for the DEM part
    self.a1[0].apply(self.init_weights)
    self.a2[0].apply(self.init_weights)
    self.a3[0].apply(self.init_weights)
    self.a4[0].apply(self.init_weights)
    
    # running the model on gpu
    self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
    #nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity="leaky_relu")
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
    
    #level 2
    a2= self.a4(self.a3(a1))
    
    #encoder visual
    #level 1
    x1 = self.c2(self.c1(input[:,1,:,:].view(input.shape[0], 1, input.shape[2], input.shape[3])))
    
    #level 2
    x2= self.c4(self.c3(x1 + a1))
    
    #level 3
    x4 = self.c5(x2 + a2)
    
    #out = torch.cat((y[:,0:2,:,:], self.sfplus(y[:,None,2,:,:]), self.sfplus(y[:,None,3,:,:])), 1)
    
    return x4


class Decoder(nn.Module):
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
    
    # necessary for all classes extending the module class
    super(Decoder, self).__init__() 
    
    # converting values inside the model into floats
    self = self.float()

    #decoder
    # the extra width is added because of concatenation ?
    self.c6 = nn.Sequential(nn.Conv2d(encoder_conv_width[4], encoder_conv_width[5], 3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[5]),nn.LeakyReLU(True))
    self.t1 = nn.Sequential(nn.ConvTranspose2d(encoder_conv_width[4], decoder_conv_width[0], 2, 2), nn.BatchNorm2d(encoder_conv_width[0]),nn.LeakyReLU(True))
    self.c7 = nn.Sequential(nn.Conv2d(decoder_conv_width[0],decoder_conv_width[0],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[0]),nn.LeakyReLU(True))
    self.c8 = nn.Sequential(nn.Conv2d(decoder_conv_width[0],decoder_conv_width[1],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[1]),nn.LeakyReLU(True))
    self.t2 = nn.Sequential(nn.ConvTranspose2d(decoder_conv_width[1], decoder_conv_width[1], 2, 2), nn.BatchNorm2d(decoder_conv_width[1]),nn.LeakyReLU(True))
    self.c9 = nn.Sequential(nn.Conv2d(decoder_conv_width[1],decoder_conv_width[2],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[2]),nn.LeakyReLU(True))
    self.c10 = nn.Sequential(nn.Conv2d(decoder_conv_width[2],decoder_conv_width[3],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[3]), nn.LeakyReLU(True)) 
    
    #final  layer
    self.final = nn.Conv2d(decoder_conv_width[3], 2, 1, padding=0, padding_mode='reflect')
    
    # initializing weights
    self.c6[0].apply(self.init_weights)
    self.t1[0].apply(self.init_weights)
    self.c7[0].apply(self.init_weights)
    self.c8[0].apply(self.init_weights)
    self.t2[0].apply(self.init_weights)
    self.c9[0].apply(self.init_weights)
    self.c10[0].apply(self.init_weights)
    self.final.apply(self.init_weights)
    
    # running the model on gpu
    self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
    #nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity="leaky_relu")
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
  def forward(self,input):
    """
    the function called to run inference
    after the model is created as an object 
    we call the method with the input as an argument
    """

    #decoder
    #level 2
    y4 = self.t1(self.c6(input))
    y3 = self.c8(self.c7(y4))
    
    #level 1       
    y2 = self.t2(y3)
    y1 = self.c10(self.c9(y2))
    out = self.final(y1)
    
    #out = torch.cat((y[:,0:2,:,:], self.sfplus(y[:,None,2,:,:]), self.sfplus(y[:,None,3,:,:])), 1)
    
    return out


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
    
    self.sigm = nn.Sigmoid()
    
    # here the convolutions don't change the width and height, only the number of channels
    self.fc1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.fc2 = nn.Sequential(nn.Conv2d(16, 8, 3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(8),nn.ReLU(True))
    self.fc3 = nn.Sequential(nn.Conv2d(8, 5, 1, padding=0, padding_mode='reflect')) #conv (1x1)

    # initiating weights
    self.fc1[0].apply(self.init_weights)
    self.fc2[0].apply(self.init_weights)
    self.fc3[0].apply(self.init_weights)
    
    self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
      nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
    
  def forward(self, x):
    """
    here x is the input
    """
    
    # applying the different layers
    x1 = self.fc3(self.fc2(self.fc1(x)))
    
    # sigmoid activation function
    out = self.sigm(x1)

    return out

def train(model, discr, optimizer_D, args, datasets):
  """
  train for one epoch
  args are some parameters of our model, e.g. batch size or n_class, etc.
  """
  
  #switch the model in training mode
  model.encoder.train()
  model.decoder.train() 
  
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
    pred = model.predict(tiles)
    code = model.encoder(tiles)
    
    # boolean matrixes to remove effect of no data
    bool_matr_alt = tiles[:,None,0,:,:] != 0
    bool_matr_rad = tiles[:,None,1,:,:] != 0
    
    # filtering the data
    pred_alt = pred[:,None,0,:,:][bool_matr_alt]
    tiles_alt = tiles[:,None,0,:,:][bool_matr_alt]
    pred_rad = pred[:,None,1,:,:][bool_matr_rad]
    tiles_rad = tiles[:,None,1,:,:][bool_matr_rad]
    
    # loading defiance matrix
    #d_mat_alt = pred[:,None,2,:,:][bool_matr_alt]
    #d_mat_rad = pred[:,None,3,:,:][bool_matr_rad]
    
    ## sum of squares
    loss_alt = loss_fun.MeanSquareError(pred_alt, tiles_alt)
    loss_rad = loss_fun.MeanSquareError(pred_rad, tiles_rad)
    
    #loss_alt = torch.mean((tiles_alt - pred_alt)**2 / (2*d_mat_alt**2+eps) + 2*torch.log(d_mat_alt+eps))
    #loss_rad = torch.mean((tiles_rad - pred_rad)**2 / (2*d_mat_rad**2+eps) + 2*torch.log(d_mat_rad+eps))
    
    # reshaping the labels 
    list_labels = [labels for i in range(code.shape[-1])]
    labels = torch.stack(list_labels, dim=-1)
    list_labels = [labels for i in range(code.shape[-1])]
    labels = torch.stack(list_labels, dim=-1)
    _, labels = labels.max(dim=1)
    
    adver = True
    
    if adver:
        
        ## now the disciminant part
        #pred_year = discr(code.detach())
        pred_year = discr(code)
        _, pred_max = pred_year.max(dim=1)
        
        ## applying loss function for the discriminator and optimizing the weights
        loss_disc = loss_fun.CrossEntropy(pred_year, labels)
        
        # optimizing the discriminator
        optimizer_D.zero_grad()
        loss_disc.backward(retain_graph=True)
        optimizer_D.step()
        
        # saving the loss
        loss_disc_val.add(loss_disc.item())
        
        # checking the accuracy
        matrix_accu = pred_max == labels
        matrix_accu_f = matrix_accu.flatten()
        matrix_accu_f = matrix_accu_f.cpu().detach().numpy()
        nb_true = np.count_nonzero(matrix_accu_f == True)
        accu_discr = nb_true / len(matrix_accu_f)
        
        # calculating the loss for the adversarial part
        pred_year = discr(model.code(tiles))
        loss_disc = loss_fun.CrossEntropy(pred_year, labels)
        
        ## optional: optimizing the encoder
        #optimizer.zero_grad()
        #loss_disc.backward(retain_graph=True)
        #optimizer.step() #one SGD step
        
    auto_encod = True
    
    if auto_encod:
        # total loss
        loss = loss_alt + loss_rad - 1 * loss_disc
        loss_data.add(loss.item())
        
        # ============backward===========
        model.opti_E.zero_grad()
        model.opti_D.zero_grad()
        loss.backward(retain_graph=True)
        model.opti_E.step() #one SGD step
        model.opti_D.step()

    # storing the loss values
    loss_data_alt.add(loss_alt.item())
    loss_data_rad.add(loss_rad.item())
    #loss_data_alt.add(loss_alt.cpu().detach())
    #loss_data_rad.add(loss_rad.cpu().detach())
    
    #for p in model.parameters(): #we clip the gradient at norm 1
    #  p.grad.data.clamp_(-1, 1) #this helps learning faster
    
    result = (loss_data.value()[0], len(loader), loss_data_alt.value()[0],
              loss_data_rad.value()[0], loss_disc_val.value()[0], accu_discr)
    
  return result


def train_full(args, datasets, writer):
  """
  The full training loop
  """
  
  #initialize the models
  encoder = Encoder(args.n_channel, args.conv_width)
  decoder = Decoder(args.n_channel, args.conv_width, args.dconv_width)
  
  discr = Discriminator()

  # total number of parameters
  print('Total number of encoder parameters: {}'.format(sum([p.numel() for p in encoder.parameters()])))
  print('Total number of encoder parameters: {}'.format(sum([p.numel() for p in decoder.parameters()])))
  
  #define the optimizer
  #adam optimizer is always a good guess for classification
  optimizer_E = optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=1e-5)
  optimizer_De = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=1e-5)
  optimizer_D = optim.Adam(discr.parameters(), lr=args.lr)
  
  # creating a model with encoder and decoder
  model = AutoEncoder(encoder, decoder, optimizer_E, optimizer_De)
  
  TRAINCOLOR = '\033[100m'
  NORMALCOLOR = '\033[0m'
  
  # storing losses to display them eventually
  losses = {"tot":[], "mns":[], "alt":[]}
  
  for i_epoch in range(args.n_epoch):
      
    #train one epoch
    loss_train, nb_batches, loss_alt, loss_rad, loss_disc, accu_discr = train(model,
                                                                              discr,
                                                                              optimizer_D,
                                                                              args,
                                                                              datasets)
    
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
    print("accu discr is %1.4f" % (accu_discr))
    
    
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
    
    writer.add_scalar('accuracy discriminator',
                    accu_discr,
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