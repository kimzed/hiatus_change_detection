# Project hiatus
# script used to do the training
# 16/11/2020
# Cédric BARON

import matplotlib.pyplot as plt
import torch
import torchnet as tnt
import numpy as np
import os
from torch.optim.lr_scheduler import MultiStepLR


os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

# importing our functions
import loss as loss_fun
import model as mod

def train(model, args, datasets, data_fusion=True, adver=True, defiance=False):
  """
  train for one epoch
  args are some parameters of our model, e.g. batch size or n_class, etc.
  """
  
  #switch the model in training mode
  model.encoder.train()
  model.decoder.train() 
  model.discr.train()

  #the loader function will take care of the batching
  # train_set was defined prior
  loader = torch.utils.data.DataLoader(datasets, \
         batch_size=args.batch_size, shuffle=True, drop_last=True)
  
  # loss on the whole dataset
  loss_data = tnt.meter.AverageValueMeter()
  loss_data_alt = tnt.meter.AverageValueMeter()
  loss_data_rad = tnt.meter.AverageValueMeter()
  loss_disc_val = tnt.meter.AverageValueMeter()
  accu_discr = 0.0
  
  # loops over the batches
  for index, (tiles, labels) in enumerate(loader):
    
    # loading on the gpu
    tiles = tiles.cuda().float()
    labels = labels.cuda().long()
    
    # ============forward===========
    # compute the prediction
    pred = model.predict(tiles, data_fusion=data_fusion, defiance=defiance)
    code = model.encoder(tiles, data_fusion=data_fusion)
    
    # boolean matrixes to remove effect of no data
    bool_matr_alt = tiles[:,None,0,:,:] != 0
    bool_matr_rad = tiles[:,None,1,:,:] != 0
    
    # filtering the data
    pred_alt = pred[:,None,0,:,:][bool_matr_alt]
    tiles_alt = tiles[:,None,0,:,:][bool_matr_alt]
    pred_rad = pred[:,None,1,:,:][bool_matr_rad]
    tiles_rad = tiles[:,None,1,:,:][bool_matr_rad]
    
    ## defiance part
    if defiance:
        # loading defiance matrix
        d_mat_alt = pred[:,None,2,:,:][bool_matr_alt]
        d_mat_rad = pred[:,None,3,:,:][bool_matr_rad]
        
        # calculating the loss
        eps = 10**-6
        loss_alt = torch.mean((tiles_alt - pred_alt)**2 / (2*d_mat_alt+eps) + (1/2)*torch.log(d_mat_alt+eps))
        loss_rad = torch.mean((tiles_rad - pred_rad)**2 / (2*d_mat_rad+eps) + (1/2)*torch.log(d_mat_rad+eps))
    else:
        ## sum of squares
        loss_alt = loss_fun.MeanSquareError(pred_alt, tiles_alt)
        loss_rad = loss_fun.MeanSquareError(pred_rad, tiles_rad)
    
    
    
    # applying arg max on labels for cross entropy
    _, labels = labels.max(dim=1)
    
    if adver:
        
        nb_loop = 1
        
        for i in range(nb_loop):
        
            ## now the disciminant part
            #pred_year = discr(code.detach())
            pred_year = model.discr(code)
            
            # applying arg max for checking accuracy
            _, pred_max = pred_year.max(dim=1)
            
            ## applying loss function for the discriminator and optimizing the weights
            loss_disc = loss_fun.CrossEntropy(pred_year, labels)
            
            # optimizing the discriminator. optional: training the encoder as well
            model.opti_D.zero_grad()
            #model.opti_AE.zero_grad()
            loss_disc.backward(retain_graph=True)
            model.opti_D.step()
            #model.opti_AE.step()
            
            # saving the loss
            loss_disc_val.add(loss_disc.item())
            
            # checking the accuracy
            matrix_accu = pred_max == labels
            matrix_accu_f = matrix_accu.flatten()
            matrix_accu_f = matrix_accu_f.cpu().detach().numpy()
            nb_true = np.count_nonzero(matrix_accu_f == True)
            accu_discr += nb_true / len(matrix_accu_f)
            
            # optional: putting an adversarial on the encoder
            #code = model.encoder(tiles)
            #pred_year = discr(code)
            #loss_disc = loss_fun.CrossEntropy(pred_year, labels)
            #loss_disc_adv = -loss_disc
            # optimizing the discriminator
            #model.opti_E.zero_grad()
            #loss_disc_adv.backward()
            #model.opti_E.step()
            
        #averaging accuracy
        accufin = accu_discr/(len(loader)*nb_loop)
        
        
        
    auto_encod = True
    
    if auto_encod:
        
        code = model.encoder(tiles, data_fusion=data_fusion)
        pred_year = model.discr(code)
        loss_disc = loss_fun.CrossEntropy(pred_year, labels)
        
        # total loss
        if adver and data_fusion:
            loss =  loss_rad + loss_alt -  1 * loss_disc   
        elif data_fusion:
            loss =  loss_rad + loss_alt
        elif adver:
            loss =  loss_rad -  1 * loss_disc
        else:
            loss = loss_rad
            
        loss_data.add(loss.item())
        
        # ============backward===========
        model.opti_AE.zero_grad()
        loss.backward()
        model.opti_AE.step()
    
    # storing the loss values
    loss_data_alt.add(loss_alt.item())
    loss_data_rad.add(loss_rad.item())
    #loss_data_alt.add(loss_alt.cpu().detach())
    #loss_data_rad.add(loss_rad.cpu().detach())
    
    if adver == False:
        accufin = 0
    #for p in model.parameters(): #we clip the gradient at norm 1
    #  p.grad.data.clamp_(-1, 1) #this helps learning faster
  
  # output of various losses
  result = (loss_data.value()[0], len(loader), loss_data_alt.value()[0],
              loss_data_rad.value()[0], loss_disc_val.value()[0], accufin)  
    
  return result


def train_full(args, datasets, writer, data_fusion=True, adver=True, defiance=False):
  """
  The full training loop
  """
  
  #initialize the models
  encoder = mod.Encoder(args.n_channel, args.conv_width)
  decoder = mod.Decoder(args.n_channel, args.conv_width, args.dconv_width)
  discr = mod.Discriminator()

  # total number of parameters
  print('Total number of encoder parameters: {}'.format(sum([p.numel() for p in encoder.parameters()])))
  print('Total number of encoder parameters: {}'.format(sum([p.numel() for p in decoder.parameters()])))
  print('Total number of discriminator parameters: {}'.format(sum([p.numel() for p in discr.parameters()])))
  
  # creating a model with encoder, decoder and discriminator
  model = mod.AdversarialAutoEncoder(encoder, decoder, discr, args.lr)
  
  # objects to update the learning rate
  scheduler_D = MultiStepLR(model.opti_D, milestones=args.lr_steps, gamma=args.lr_decay)
  scheduler_AE = MultiStepLR(model.opti_AE, milestones=args.lr_steps, gamma=args.lr_decay)
  
  TRAINCOLOR = '\033[100m'
  NORMALCOLOR = '\033[0m'
  
  # storing losses to display them eventually
  losses = {"tot":[], "mns":[], "alt":[], "accu":[]}
  
  for i_epoch in range(args.n_epoch):
      
    #train one epoch
    loss_train, nb_batches, loss_alt, loss_rad, loss_disc, accu_discr = train(model,
                                                                              args,
                                                                              datasets,
                                                                              data_fusion=data_fusion,
                                                                              adver=adver,
                                                                              defiance=defiance)
    
    # updating the learning rate
    scheduler_D.step()
    scheduler_AE.step()
    
    # storing loss for later plotting
    losses["tot"].append(loss_train)
    losses["mns"].append(loss_alt)
    losses["alt"].append(loss_rad)
    losses["accu"].append(accu_discr)
    
    print(TRAINCOLOR)
    print('Epoch %3d -> Train Loss: %1.4f' % (i_epoch, loss_train) + NORMALCOLOR)
    print("loss mns is %1.4f" % (loss_alt))
    print("loss rad is %1.4f" % (loss_rad))
    print("loss discr is %1.4f" % (loss_disc))
    print("accu discr is %1.4f" % (accu_discr))
    
    # ...log the running loss
    writer.add_scalar('training loss',
                    loss_train,
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
  
  plt.title('accuracy of the discriminator')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.plot(range(len(losses["accu"])), losses["accu"])
  plt.show()
  
  return model