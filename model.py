# Project hiatus
# model
# 02/11/2020
# Cédric BARON

import torch.nn as nn
import torch
import torch.optim as optim


class AdversarialAutoEncoder(nn.Module):
  """
  EncoderDecoder network for semantic segmentation
  """
  
  def __init__(self, encoder, decoder, discr=0, learning_rate=0.01):
      
      super(AdversarialAutoEncoder, self).__init__()
      
      # saving the two models in the object
      self.encoder = encoder
      self.decoder = decoder
      if discr:
          self.discr = discr
      
      # combining parameters
      self.AE_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
      
      # saving the optimizers in the object
      self.opti_AE =  optim.Adam(self.AE_params, learning_rate, weight_decay=0)
      if discr:
          self.opti_D =  optim.Adam(self.discr.parameters(), 0.002, weight_decay=1e-5)
      
  def predict(self, input, args):
      
      # compute code and output
      code = self.encoder(input, args)
      out = self.decoder(code, args)
      
      return out
  
    
class Encoder(nn.Module):
  """
  EncoderDecoder network for semantic segmentation
  """
  
  def __init__(self, encoder_conv_width, args):
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
    self.c1_rad = nn.Sequential(nn.Conv2d(1, encoder_conv_width[0],kernel_size=3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[0]),nn.LeakyReLU(True))
    self.sc2_rad = nn.Sequential(nn.Conv2d(encoder_conv_width[0],encoder_conv_width[1],kernel_size=4,padding=1, stride=2, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[1]),nn.LeakyReLU(True))
    self.c3 = nn.Sequential(nn.Conv2d(encoder_conv_width[1],encoder_conv_width[2],kernel_size=3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[2]),nn.LeakyReLU(True))
    self.sc4 = nn.Sequential(nn.Conv2d(encoder_conv_width[2],encoder_conv_width[3],kernel_size=4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[3]),nn.LeakyReLU(True))
    self.c5 = nn.Sequential(nn.Conv2d(encoder_conv_width[3],encoder_conv_width[4],kernel_size=3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[4]),nn.LeakyReLU(True))
   
    # network for the altitude
    self.c1_dem = nn.Sequential(nn.Conv2d(1, encoder_conv_width[0],3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[0]),nn.LeakyReLU(True))
    self.sc2_dem = nn.Sequential(nn.Conv2d(encoder_conv_width[0],encoder_conv_width[1],kernel_size=4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[1]),nn.LeakyReLU(True))
    self.c3_dem = nn.Sequential(nn.Conv2d(encoder_conv_width[1],encoder_conv_width[2],kernel_size=3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[2]),nn.LeakyReLU(True))
    self.sc4_dem = nn.Sequential(nn.Conv2d(encoder_conv_width[2],encoder_conv_width[3],kernel_size=4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(encoder_conv_width[3]),nn.LeakyReLU(True))

    #weight initialization
    self.c1_rad[0].apply(self.init_weights)
    self.sc2_rad[0].apply(self.init_weights)
    self.c3[0].apply(self.init_weights)
    self.sc4[0].apply(self.init_weights)
    self.c5[0].apply(self.init_weights)
    
    # for the DEM part
    self.c1_dem[0].apply(self.init_weights)
    self.sc2_dem[0].apply(self.init_weights)
    self.c3_dem[0].apply(self.init_weights)
    self.sc4_dem[0].apply(self.init_weights)
    
    if args.cuda:
        # running the model on gpu
        self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
    #nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity="leaky_relu")
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
    
  def forward(self, input, args):
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
    if args.rad_input:
        a1 = self.sc2_dem(self.c1_dem(alt))
    else:
        a1 = self.sc2_dem(self.c1_dem(rad))
        
    #level 2
    a2= self.sc4_dem(self.c3_dem(a1))
    
    #encoder visual
    #level 1
    if args.rad_input:
        x1 = self.sc2_rad(self.c1_rad(rad))
    else:
        x1 = self.sc2_rad(self.c1_rad(alt))
    
    if args.data_fusion:
        #level 2
        x2= self.sc4(self.c3(x1 + a1))
        
        #level 3
        x3 = self.c5(x2 + a2)
        
    else:
        #level 2
        x2= self.sc4(self.c3(x1))
        
        #level 3
        x3 = self.c5(x2)
        
    return x3


class Decoder(nn.Module):
  """
  EncoderDecoder network for semantic segmentation
  """
  
  def __init__(self, encoder_conv_width, decoder_conv_width, args):
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
    self.c6 = nn.Sequential(nn.Conv2d(encoder_conv_width[4], decoder_conv_width[0], kernel_size=3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[0]),nn.LeakyReLU(True))
    self.t1 = nn.Sequential(nn.ConvTranspose2d(decoder_conv_width[0], decoder_conv_width[1], kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(decoder_conv_width[1]),nn.LeakyReLU(True))
    self.c7 = nn.Sequential(nn.Conv2d(decoder_conv_width[1],decoder_conv_width[1],kernel_size=3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[1]),nn.LeakyReLU(True))
    self.c8 = nn.Sequential(nn.Conv2d(decoder_conv_width[1],decoder_conv_width[2],kernel_size=3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[2]),nn.LeakyReLU(True))
    self.t2 = nn.Sequential(nn.ConvTranspose2d(decoder_conv_width[2], decoder_conv_width[2], kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(decoder_conv_width[2]),nn.LeakyReLU(True))
    self.c9 = nn.Sequential(nn.Conv2d(decoder_conv_width[2],decoder_conv_width[3],kernel_size=3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[3]),nn.LeakyReLU(True))
    self.c10 = nn.Sequential(nn.Conv2d(decoder_conv_width[3],decoder_conv_width[4],kernel_size=3,padding=1, padding_mode='reflect'),nn.BatchNorm2d(decoder_conv_width[4]), nn.LeakyReLU(True)) 
    
    if args.defiance:
          # extra convs defiance
          self.c11 = nn.Sequential(nn.Conv2d(args.dconv_width[4],args.def_width[0],kernel_size=1),nn.BatchNorm2d(args.def_width[0]), nn.LeakyReLU(True)) 
          self.c12 = nn.Sequential(nn.Conv2d(args.def_width[0],args.def_width[1],kernel_size=1),nn.BatchNorm2d(args.def_width[1]), nn.LeakyReLU(True)) 
          self.c13 = nn.Sequential(nn.Conv2d(args.def_width[1],args.def_width[2],kernel_size=1),nn.BatchNorm2d(args.def_width[2]), nn.LeakyReLU(True)) 
          self.c14 = nn.Sequential(nn.Conv2d(args.def_width[2],args.def_width[3],kernel_size=1),nn.BatchNorm2d(args.def_width[3]), nn.LeakyReLU(True)) 
          self.c15 = nn.Sequential(nn.Conv2d(args.def_width[3],args.def_width[4],kernel_size=1),nn.BatchNorm2d(args.def_width[4]), nn.LeakyReLU(True)) 
          self.defi = nn.Conv2d(16, 1, 1, padding=0)  
          
    # aleotoric part
    self.final = nn.Conv2d(decoder_conv_width[4], 2, 1, padding=0)
    
    # initializing weights
    self.c6[0].apply(self.init_weights)
    self.t1[0].apply(self.init_weights)
    self.t2[0].apply(self.init_weights)
    self.c7[0].apply(self.init_weights)
    self.c8[0].apply(self.init_weights)
    self.c9[0].apply(self.init_weights)
    self.c10[0].apply(self.init_weights)
    self.final.apply(self.init_weights)
    
    if args.defiance:
        # initializing weights
        self.c11[0].apply(self.init_defiance)
        self.c12[0].apply(self.init_defiance)
        self.c13[0].apply(self.init_defiance)
        self.c14[0].apply(self.init_defiance)
        self.c15[0].apply(self.init_defiance)
        self.defi.apply(self.init_defiance_final)
    
    # running the model on gpu
    if args.cuda:
        self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
      nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
    
  def init_defiance(self, layer):
      layer.weight.data.normal_(0, 0.000001)
      layer.bias.data.fill_(0) 
      
  def init_defiance_final(self, layer):
      layer.weight.data.normal_(0, 0.000001)
      layer.bias.data.fill_(torch.tensor(0.5413)) 
      
      
      
  def forward(self,input, args):
    """
    the function called to run inference
    after the model is created as an object 
    we call the method with the input as an argument
    """

    #decoder
    #level 2
    y4 = nn.Upsample(scale_factor=2, mode='bilinear')(self.c6(input))
    y3 = self.c8(self.c7(y4))
    
    #level 1   
    y2 = self.c9(nn.Upsample(scale_factor=2, mode='bilinear')(y3))
    y1 = self.c10(y2)
    
    out = self.final(y1)
    
    if not args.data_fusion:
        # adding a matrix of zeros for dem
        none_mat = torch.zeros(out.shape[0], 1, out.shape[2], out.shape[3])
        none_mat = none_mat.cuda()
        
        if args.rad_input:
            out[:,0,:,:][:,None,:,:] = none_mat
        else:
            out[:,1,:,:][:,None,:,:] = none_mat
    
    if args.defiance:
        # including defiance
        defiance_rad = self.c15(self.c14(self.c13(self.c12(self.c11(y1)))))
        aleo_final = torch.nn.Softplus()(self.defi(defiance_rad))
        out = torch.cat((out[:,0:2,:,:], aleo_final), 1)
        
    return out


class Discriminator(nn.Module):
  """
  #Discriminator network for year detection
  """
  
  def __init__(self, args):
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
    if args.split:
        self.sc1 = nn.Sequential(nn.Conv2d(args.nb_channels_split, args.disc_width[1], 4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    else:
        self.sc1 = nn.Sequential(nn.Conv2d(args.conv_width[-1], args.disc_width[1], 4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.c1 = nn.Sequential(nn.Conv2d(args.disc_width[1], args.disc_width[2], 3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.sc2 = nn.Sequential(nn.Conv2d(args.disc_width[2], args.disc_width[3], 4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.c2 = nn.Sequential(nn.Conv2d(args.disc_width[3], args.disc_width[4], 3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.sc3 = nn.Sequential(nn.Conv2d(args.disc_width[4], args.disc_width[5], 4, stride=2, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    self.c3 = nn.Sequential(nn.Conv2d(args.disc_width[5], args.disc_width[6], 3, padding=1, padding_mode='reflect'),nn.BatchNorm2d(16),nn.ReLU(True))
    
    # FC layers
    self.lin = nn.Sequential(nn.Linear(args.disc_width[7], args.disc_width[8]),nn.BatchNorm1d(args.disc_width[8]),nn.ReLU(True))
    self.lin2 = nn.Linear(args.disc_width[8], 5)
    
    # initiating weights
    self.sc1[0].apply(self.init_weights)
    self.c1[0].apply(self.init_weights)
    self.sc2[0].apply(self.init_weights)
    self.c2[0].apply(self.init_weights)
    self.sc3[0].apply(self.init_weights)
    self.c3[0].apply(self.init_weights)
    self.lin[0].apply(self.init_weights)
    self.lin2.apply(self.init_weights)
    
    # running on gpu
    if args.cuda:
        self.cuda()
    
  def init_weights(self,layer): #gaussian init for the conv layers
      nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    

    
  def forward(self, code, args):
    """
    here x is the input
    """
    if args.split:
        # splitting the code
        code = code[:,:args.nb_channels_split,:,:]
    
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
