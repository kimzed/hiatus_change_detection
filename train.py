# Project hiatus
# script used to do the training
# 16/11/2020
# Cédric BARON

import torch
import torchnet as tnt
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
from sklearn import metrics
import datetime
from torch.utils.tensorboard import SummaryWriter
import os

# importing our functions
import loss as loss_fun
import model as mod
import evaluate as eval_model
import utils as fun

def train(model, args, datasets):
  """
  train for one epoch
  args are some parameters of our model, e.g. batch size or n_class, etc.
  """
  
  #switch the model in training mode
  model.encoder.train()
  model.decoder.train() 
  if args.adversarial:
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
    if args.cuda:
        tiles = tiles.cuda().float()
        labels = labels.cuda().long()
    else:
        tiles = tiles.float()
        labels = labels.long()
    
    # adding noise to the sample
    noise = np.random.normal(0, 0.01, tiles.shape)
    noise_tens = fun.torch_raster(noise)
    
    # adding noise
    tiles_noise = tiles + noise_tens
    
   
    # applying arg max on labels for cross entropy
    _, labels = labels.max(dim=1)
    
    # ============discriminator===========
    
    if args.adversarial:
        
        # ============forward===========
        
        #pred_year = discr(code.detach())
        
        code = model.encoder(tiles_noise, args)
        pred_year = model.discr(code, args)
        
        # ============loss===========
        
        # applying arg max for checking accuracy
        _, pred_max = pred_year.max(dim=1)
        
        ## applying loss function for the discriminator and optimizing the weights
        loss_disc = loss_fun.CrossEntropy(pred_year, labels)
        
        
        # checking the accuracy
        matrix_accu = pred_max == labels
        matrix_accu_f = matrix_accu.flatten()
        matrix_accu_f = matrix_accu_f.cpu().detach().numpy()
        nb_true = np.count_nonzero(matrix_accu_f == True)
        accu_discr += nb_true / len(matrix_accu_f)
        
        # ============backward===========
        
        # optimizing the discriminator. optional: training the encoder as well
        model.opti_D.zero_grad()
        #model.opti_AE.zero_grad()
        loss_disc.backward(retain_graph=True)
        
        #we clip the gradient at norm 1 this helps learning faster
        if args.grad_clip:
            for p in model.discr.parameters():
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
                
        model.opti_D.step()
        model.opti_AE.zero_grad()
        model.opti_AE.step()
    
        # saving the loss
        loss_disc_val.add(loss_disc.item())
        
        # putting an adversarial training on the encoder
        if args.opti_adversarial_encoder:
            code = model.encoder(tiles, args)
            pred_year = model.discr(code, args)
            loss_disc = loss_fun.CrossEntropy(pred_year, labels)
            loss_disc_adv = loss_disc
            model.opti_AE.zero_grad()
            loss_disc_adv.backward()
            model.opti_AE.step()
        
        #averaging accuracy
        accufin = accu_discr/(len(loader))
        
        
        
    
    # ============auto_encoder optimization===========
    
    # ============forward auto-encoder===========
    # compute the prediction
    pred = model.predict(tiles_noise, args)
    code = model.encoder(tiles_noise, args)
    
    # boolean matrixes to remove effect of no data
    bool_matr_alt = tiles[:,None,0,:,:] != 0
    bool_matr_rad = tiles[:,None,1,:,:] != 0
    
    # filtering the data
    pred_alt = pred[:,None,0,:,:][bool_matr_alt]
    tiles_alt = tiles[:,None,0,:,:][bool_matr_alt]
    pred_rad = pred[:,None,1,:,:][bool_matr_rad]
    tiles_rad = tiles[:,None,1,:,:][bool_matr_rad]
    
    ## defiance part
    if args.defiance:
        # loading defiance matrix
        d_mat_rad = pred[:,None,2,:,:][bool_matr_rad]
        
        # calculating the loss
        eps = 10**-5
        loss_alt = loss_fun.MeanSquareError(pred_alt, tiles_alt)
        
        # loss for the defiance
        mse_rad = (tiles_rad - pred_rad)**2
        loss_rad = torch.mean(mse_rad / (d_mat_rad+eps) + (1/2)*torch.log(d_mat_rad+eps))
 
        
    else:
        ## sum of squares
        loss_alt = loss_fun.MeanSquareError(pred_alt, tiles_alt)
        loss_rad = loss_fun.MeanSquareError(pred_rad, tiles_rad)
    
    
    if args.auto_encod:
        
        # ============forward===========
        if args.adversarial:
            code = model.encoder(tiles_noise, args)
            pred_year = model.discr(code, args)
            loss_disc = loss_fun.CrossEntropy(pred_year, labels)
            
        # ============loss==========
        
        if args.adversarial and args.data_fusion:
            loss =  loss_rad + loss_alt #-  args.disc_loss_weight * loss_disc   
        elif args.data_fusion:
            loss =  loss_rad + loss_alt
        elif args.adversarial and args.rad_input:
            loss =  loss_rad -  args.disc_loss_weight * loss_disc
        elif args.adversarial:
            loss =  loss_alt -  args.disc_loss_weight * loss_disc
        elif args.rad_input:
            loss = loss_rad
        else:
            loss = loss_alt
            
        loss_data.add(loss.item())
        
        # ============backward===========
        
            
        model.opti_AE.zero_grad()
        loss.backward()
        
        #we clip the gradient at norm 1 this helps learning faster
        if args.grad_clip:
            for p in model.AE_params:
                p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
            
        model.opti_AE.step()
        
    
    # storing the loss values
    loss_data_alt.add(loss_alt.item())
    loss_data_rad.add(loss_rad.item())
    
    if args.adversarial == False:
        accufin = 0
  
  
  # output of various losses
  result = (loss_data.value()[0], len(loader), loss_data_alt.value()[0],
              loss_data_rad.value()[0], loss_disc_val.value()[0], accufin)
  
  return result


def train_full(args, datasets, gt_change, dict_model=None):
  """
  The full training loop
  """
  
  # get the time of the run to save the model
  now = datetime.datetime.now()
  now = now.strftime("%Y-%m-%d %H:%M")
  now = str(now)
  
  ## working with tensorboard
  writer = SummaryWriter('runs/'+now)
  
  # loading a previous model as an option
  if dict_model:
      # creating a model with encoder, decoder and discriminator
      model = fun.load_model_from_dict(dict_model)
      print(model)
  else:
      #initialize the models
      encoder = mod.Encoder(args.conv_width, args)
      decoder = mod.Decoder(args.conv_width, args.dconv_width, args)
      
      if args.adversarial:
          discr = mod.Discriminator(args)
      else:
          discr = 0
    
      # total number of parameters
      print('Total number of encoder parameters: {}'.format(sum([p.numel() for p in encoder.parameters()])))
      print('Total number of decoder parameters: {}'.format(sum([p.numel() for p in decoder.parameters()])))
      if args.adversarial:
          print('Total number of discriminator parameters: {}'.format(sum([p.numel() for p in discr.parameters()])))
      
      # creating a model with encoder, decoder and discriminator
      model = mod.AdversarialAutoEncoder(encoder, decoder, discr, args.lr)
      print(model)
  
  # objects to update the learning rate
  if args.adversarial:
      scheduler_D = MultiStepLR(model.opti_D, milestones=args.lr_steps, gamma=args.lr_decay)
  scheduler_AE = MultiStepLR(model.opti_AE, milestones=args.lr_steps, gamma=args.lr_decay)
  
  # storing losses to display them eventually
  losses = {"tot":[], "mns":[], "alt":[], "accu":[], "auc":[]}
  
  for i_epoch in range(args.epochs):
      
    #train one epoch
    loss_train, nb_batches, loss_alt, loss_rad, loss_disc, accu_discr = train(model,
                                                                              args,
                                                                              datasets)
    
    if args.test_auc:
        ## checking the auc
        model.encoder.eval()
        
        ## evaluating the model
        pred, y, classes = eval_model.generate_prediction_model(gt_change, model, args)
        
        # computing the auc
        auc = metrics.roc_auc_score(y, pred)
    else:
        auc=0
    
    
    # updating the learning rate
    if args.adversarial:
        scheduler_D.step()
    scheduler_AE.step()
    
    # storing loss for later plotting
    losses["tot"].append(loss_train)
    losses["mns"].append(loss_alt)
    losses["alt"].append(loss_rad)
    losses["accu"].append(accu_discr)
    losses["auc"].append(auc)
    
    print('Epoch %3d -> Train Loss: %1.4f' % (i_epoch, loss_train))
    print("loss mns is %1.4f" % (loss_alt))
    print("loss rad is %1.4f" % (loss_rad))
    print("loss discr is %1.4f" % (loss_disc))
    print("accu discr is %1.4f" % (accu_discr))
    print("auc is %1.4f" % (auc))
    print("\n")
    
    
    ## we save each epoch
    # save the model
    if args.load_best_model:
          
        # save the module
        torch.save(model.state_dict(), "evaluation_models/"+"save_model_epoch_"+str(i_epoch))#+now)
        
        # save a text file with the parameters of the module
        f = open("evaluation_models/"+"save_model_epoch_"+str(i_epoch)+".txt", "w")
        f.write(str(args))
        f.close()
    
    # saving the running loss
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
  
  # save the best model
  if args.load_best_model:
      
      # loading the best model
      i_best_model = losses["auc"].index(max(losses["auc"]))
      name_model = "evaluation_models/save_model_epoch_" + str(i_best_model)
      name_args = "evaluation_models/save_model_epoch_" + str(i_best_model) + ".txt"
      model, args = fun.load_model(name_model, name_args)
      
      if args.save:
          # save the model
          torch.save({'epoch': i_best_model + 1, 'args': args, 'state_dict': model.state_dict()}, os.path.join(args.output_dir+args.name_model))
          
          ## deleting the epochs saves
          # getting the list of files and adding the directory
          list_files = os.listdir("evaluation_models/")
          list_files = ["evaluation_models/"+file for file in list_files]
          # removing the files
          [os.remove(file) for file in list_files if "save_model_epoch_" in file]
      else:
          ## deleting the epochs saves
          # getting the list of files and adding the directory
          list_files = os.listdir("evaluation_models/")
          list_files = ["evaluation_models/"+file for file in list_files]
          # removing the files
          [os.remove(file) for file in list_files if "save_model_epoch_" in file]
          
  # save the model
  elif args.save:
      
      # save the model
      torch.save({'epoch': i_epoch + 1, 'args': args, 'state_dict': model.state_dict()}, os.path.join(args.output_dir+args.name_model))
      
  # visualize losses from the model
  fun.visu_losses_model(losses)
  
  # getting into eval() mode
  model.encoder.eval()
  model.decoder.eval()
  if args.adversarial:
      model.discr.eval()
  
  return model


def train_alternative_model(model1, model2, args, datasets):
  """
  function train for the
  """
  
  # loading data per year
  years = list(datasets.keys())
  data_year1 = datasets[years[0]]
  data_year2 = datasets[years[1]]
  
  #the loader function will take care of the batching
  # train_set was defined prior
  loader1 = torch.utils.data.DataLoader(data_year1, \
         batch_size=args.batch_size, shuffle=False, drop_last=True)
  loader2 = torch.utils.data.DataLoader(data_year2, \
         batch_size=args.batch_size, shuffle=False, drop_last=True)
  
  #switch the model in training mode
  model1.encoder.train()
  model1.decoder.train() 
  model2.encoder.train()
  model2.decoder.train() 
  
  # loss on the whole dataset
  loss_data = tnt.meter.AverageValueMeter()
  
  # loops over the batches
  for year1, year2 in zip(loader1, loader2):
    
    # loading on the gpu
    if args.cuda:
        year1 = year1.cuda().float()
        year2 = year2.cuda().float()
    else:
        year1 = year1.float()
        year2 = year2.float()
    
    
    # ============forward auto-encoder===========
    
    # loading the codes
    code1 = model1.encoder(year1, args)
    code2 = model2.encoder(year2, args)
    
    # computing the first loss
    loss_codes = loss_fun.MeanSquareError(code1, code2)
    
    # computing the reconstitution
    pred_year2 = model1.decoder(code1, args)
    pred_year1 = model2.decoder(code2, args)
    
    # boolean matrixes to remove effect of no data
    bool_matr1 = year1 != 0
    bool_matr2 = year2 != 0
    
    # filtering the data
    pred_year1 = pred_year1[bool_matr1]
    pred_year2 = pred_year2[bool_matr2]
    year1 = year1[bool_matr1]
    year2 = year2[bool_matr2]
    
    # computing the reconstitution losses
    loss2 = loss_fun.MeanSquareError(year2, pred_year2)
    loss3 = loss_fun.MeanSquareError(year1, pred_year1)
    
    # adding into total loss
    loss_total = loss_codes + loss2 + loss3
    
    # backpropagation
    model1.opti_AE.zero_grad()
    model2.opti_AE.zero_grad()
    loss_total.backward()
    
    # optimization
    model1.opti_AE.step()
    model2.opti_AE.step()
    
    #we clip the gradient at norm 1 this helps learning faster
    if args.grad_clip:
        for p in model1.AE_params:
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
        for p in model2.AE_params:
            p.register_hook(lambda grad: torch.clamp(grad, -1, 1))
            
    loss_data.add(loss_total.item())
  
  return loss_data.value()[0]


def train_full_alternative_model(args, datasets, dict_model):
  """
  The full training loop for the alternative model
  """
  
  # creating a model with encoder, decoder and discriminator
  model1 = fun.load_model_from_dict(dict_model)
  model2 = fun.load_model_from_dict(dict_model)
  
  # objects to update the learning rate
  scheduler_AE1 = MultiStepLR(model1.opti_AE, milestones=args.lr_steps, gamma=args.lr_decay)
  scheduler_AE2 = MultiStepLR(model2.opti_AE, milestones=args.lr_steps, gamma=args.lr_decay)
  
  # storing losses to display them eventually
  losses = {"tot":[], "mns":[], "alt":[]}
  
  for i_epoch in range(args.epochs):
      
    #train one epoch
    loss_train = train_alternative_model(model1, model2, args, datasets)
    
    # updating the learning rate
    scheduler_AE1.step()
    scheduler_AE2.step()
    
    # storing loss for later plotting
    losses["tot"].append(loss_train)
    
    print('Epoch %3d -> Train Loss: %1.4f' % (i_epoch, loss_train))
    print("\n")
    
  # save the model
  if args.save:
      
      # save the model
      torch.save({'epoch': i_epoch + 1, 'args': args, 'state_dict': model1.state_dict()}, os.path.join(args.output_dir+args.name_model+"_1"))
      torch.save({'epoch': i_epoch + 1, 'args': args, 'state_dict': model2.state_dict()}, os.path.join(args.output_dir+args.name_model+"_2"))
      
  # visualize losses from the model
  fun.visu_losses_model(losses)
  
