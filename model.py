# Project hiatus
# model
# 02/11/2020
# Cédric BARON

import torch.nn as nn
import os

os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

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
    
    # load altitude and reshape it
    alt = input[:,0,:,:]
    alt = alt.view(input.shape[0], 1, input.shape[2], input.shape[3])
    
    # load rad and reshape it
    rad = input[:,1,:,:]
    rad = rad.view(input.shape[0], 1, input.shape[2], input.shape[3])
    
    #encoder altitude
    #level 1
    a1 = self.a2(self.a1(alt))
    
    #level 2
    a2= self.a4(self.a3(a1))
    
    #encoder visual
    #level 1
    x1 = self.c2(self.c1(rad))
    
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
