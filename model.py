# Project hiatus
# model
# 02/11/2020
# Cédric BARON

import torch.nn as nn
import os
import torch
import torch.optim as optim

os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

class AdversarialAutoEncoder(nn.Module):
  """
  EncoderDecoder network for semantic segmentation
  """
  
  def __init__(self, encoder, decoder, discr, learning_rate):
      
      super(AdversarialAutoEncoder, self).__init__()
      
      # saving the two models in the object
      self.encoder = encoder
      self.decoder = decoder
      self.discr = discr
      
      # combining parameters
      AE_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
      
      # saving the optimizers in the object
      self.opti_AE =  optim.Adam(AE_params, learning_rate, weight_decay=1e-5)
      self.opti_D =  optim.Adam(self.discr.parameters(), learning_rate, weight_decay=1e-5)
      
      
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
    self.sc2 = nn.Sequential(nn.Conv2d(encoder_conv_width[0],encoder_conv_width[1],4,padding=1, stride=2, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[1]),nn.LeakyReLU(True))
    self.c3 = nn.Sequential(nn.Conv2d(encoder_conv_width[1],encoder_conv_width[2],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[2]),nn.LeakyReLU(True))
    self.sc4 = nn.Sequential(nn.Conv2d(encoder_conv_width[2],encoder_conv_width[3],4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[3]),nn.LeakyReLU(True))
    self.c5 = nn.Sequential(nn.Conv2d(encoder_conv_width[3],encoder_conv_width[4],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[4]),nn.LeakyReLU(True))
    
    # network for the altitude
    self.ca1 = nn.Sequential(nn.Conv2d(n_channels, encoder_conv_width[0],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[0]),nn.LeakyReLU(True))
    self.sca2 = nn.Sequential(nn.Conv2d(encoder_conv_width[0],encoder_conv_width[1],4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[1]),nn.LeakyReLU(True))
    self.ca3 = nn.Sequential(nn.Conv2d(encoder_conv_width[1],encoder_conv_width[2],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[2]),nn.LeakyReLU(True))
    self.sca4 = nn.Sequential(nn.Conv2d(encoder_conv_width[2],encoder_conv_width[3],4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[3]),nn.LeakyReLU(True))

    #weight initialization
    self.c1[0].apply(self.init_weights)
    self.sc2[0].apply(self.init_weights)
    self.ca3[0].apply(self.init_weights)
    self.sc4[0].apply(self.init_weights)
    self.c5[0].apply(self.init_weights)
    
    # for the DEM part
    self.ca1[0].apply(self.init_weights)
    self.sca2[0].apply(self.init_weights)
    self.c3[0].apply(self.init_weights)
    self.sca4[0].apply(self.init_weights)
    
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
    
    # load altitude and reshape it
    alt = input[:,0,:,:][:,None,:,:]
    
    # load rad and reshape it
    rad = input[:,1,:,:][:,None,:,:]
    
    #encoder altitude
    #level 1
    a1 = self.sca2(self.ca1(alt))
    
    #level 2
    a2= self.sca4(self.ca3(a1))
    
    #encoder visual
    #level 1
    x1 = self.sc2(self.c1(rad))
    
    #level 2
    x2= self.sc4(self.c3(x1 + a1))
    
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
    
    self.softm = nn.Softmax(dim=1)
    
    # convolution steps
    self.sc1 = nn.Sequential(nn.Conv2d(16, 16, 4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.c1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.sc2 = nn.Sequential(nn.Conv2d(16, 16, 4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.c2 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.sc3 = nn.Sequential(nn.Conv2d(16, 16, 4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.c3 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    
    # FC layers
    self.lin = nn.Sequential(nn.Linear(16, 16),nn.BatchNorm1d(16),nn.ReLU(True))
    self.lin2 = nn.Sequential(nn.Linear(16, 5),nn.BatchNorm1d(5),nn.ReLU(True))
    
    # initiating weights
    self.sc1[0].apply(self.init_weights)
    self.c1[0].apply(self.init_weights)
    self.sc2[0].apply(self.init_weights)
    self.c2[0].apply(self.init_weights)
    self.sc3[0].apply(self.init_weights)
    self.c3[0].apply(self.init_weights)
    self.lin[0].apply(self.init_weights)
    self.lin2[0].apply(self.init_weights)
    
    self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
      nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
    
  def forward(self, code):
    """
    here x is the input
    """
    
    # applying the convolution layers
    x1 = self.c1(self.sc1(code))
    x2 = self.c2(self.sc2(x1))
    x3 = self.c3(self.sc3(x2))
    
    # performing a mean pool
    m1 = torch.mean(x3, dim=-1)
    m2 = torch.mean(m1, dim=-1)
    
    # FC layers
    x6 = self.lin2(self.lin(m2))
    
    # sigmoid activation function
    out = self.softm(x6)

    return out



















