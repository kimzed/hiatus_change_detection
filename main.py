# Project hiatus
# main script with a parser for the model
# 12/10/2020
# Cédric BARON

# loading required packages
import torch
import random
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import argparse
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# for manual visualisation
from rasterio.plot import show

# putting the right work directory
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

# importing our functions
import utils as fun
import train as train
import evaluate as eval_model
import metrics as fun_metrics

def main():
    
    # create the parser with all arguments
    parser = argparse.ArgumentParser(description='Auto-encoder Time Adversarial Model')
    
    # Optimization arguments
    parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default='[50, 70, 90]', help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=92, type=int, help='Batch size')
    parser.add_argument('--optim', default='adam', help='Optimizer: sgd|adam')
    parser.add_argument('--grad_clip', default=1, type=float, help='Element-wise clipping of gradient. If 0, does not clip')
    
    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--test_nth_epoch', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--save_nth_epoch', default=1, type=int, help='Save model each n-th epoch during training')

    # Dataset
    parser.add_argument('--dataset', default='frejus_dataset', help='Dataset name: frejus_dataset')
    
    # Model
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')
    parser.add_argument('--data_fusion', default=True, help='Including data fusion')
    parser.add_argument('--adversarial', default=True, help='Making the model adversarial')
    parser.add_argument('--defiance', default=False, help='Including defiance')
    parser.add_argument('--split', default=True, help='Making a split on the code')
    parser.add_argument('--auto_encod', default=True, help='Activating the auto-encoder')
    
    # Encoder
    parser.add_argument('--conv_width', default=[8,8,16,16,16,16], help='Layers size')
    
    # Decoder
    parser.add_argument('--dconv_width', default=[8,8,8,8], help='Layers size')
    
    # Discriminator
    parser.add_argument('--nb_trains_discr', default=1, type=int, help='Number of times the discriminator is trained compared to the autoencoder')
    parser.add_argument('--disc_loss_weight', default=0.1, type=float, help='Weight applied on the adversarial loss with full model')
    parser.add_argument('--opti_adversarial_encoder', default=False, help='Trains the encoder weights')
    
    args = parser.parse_args()
    args.start_epoch = 0
    
    # setting the seed
    set_seed(args.seed, args.cuda)
    
    # Decide on the dataset
    if args.dataset=='frejus_dataset':
        
        # loading the dataset, getting a raster for later data visualisation
        # after every epoch
        import frejus_dataset
        train_data, gt_change, numpy_rasters, ex_raster = frejus_dataset.get_datasets(ex_raster=True)
        
    ## working with tensorboard
    writer = SummaryWriter('runs/0212_1_test')
    
    # training the model
    trained_model = train.train_full(args, train_data, writer, gt_change,
                                                    ex_raster)
    
    return args, gt_change, numpy_rasters, trained_model, train_data


def set_seed(seed, cuda=True):
    """ 
    Sets seeds in all frameworks
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if cuda: 
        torch.cuda.manual_seed(seed)  
        
###############################################################################
###############################################################################
###############################################################################
        
if __name__ == "__main__": 
    
    print(
    """
    Training the model and loading the data
    """)
    
    # running the model
    args, gt_change, numpy_rasters, trained_model, datasets = main()
    
    # removing the year vector from the data
    datasets = [data[0] for data in datasets]
    
if __name__ == "__other__": 
    
    
    print(
    """
    Visualizing some predictions for the autoencoder
    """)
    
    for i in range(5):
        
        # visualizing training raster
        raster = datasets[i]
        fun.visualize(raster, third_dim=False)
        
        # visualizing prediction
        pred = trained_model.predict(raster[None,:,:,:].float().cuda(), data_fusion=args.data_fusion)[0].cpu()
        fun.visualize(pred.detach().numpy().squeeze(), third_dim=False, defiance=args.defiance)
       
    print(
    '''
    Now we are going to visualize various embeddings in the model itself
    ''')
    
    # visualizing for a random index number the inner embeddings
    fun.view_u(datasets, trained_model, args, random.randint(0, 900))
    
    # visualizing embedding inside the model
    nb = random.randint(0, 900)
    fun.view_u(numpy_rasters["1966"], trained_model, args, nb)
    fun.view_u(numpy_rasters["1970"], trained_model, args, nb)
    
    print(
    """
    Performing change detection analysis on actual data
    """)
    
    ind = random.randint(0, 900)
    
    # interesting nbs 54-70: 783, 439,746, 201 706 715
    # no change 66-70: 245 799 406 437, 715
    
    ind = random.randint(0, 900)
    print(ind)
    fun.visualize(numpy_rasters["1954"][ind][:,:,:], third_dim=False)
    fun.visualize(numpy_rasters["1989"][ind][:,:,:], third_dim=False)
    
    nb = 715
    ind = nb
    ## running cd model
    rast1 = numpy_rasters["1954"][nb][None,:,:,:]
    rast2 = numpy_rasters["1970"][nb][None,:,:,:]
    
    threshold = 0.5
    
    # computing change raster
    cmap, dccode, code1, code2 = fun.change_detection(rast1, rast2, trained_model, args,
                                                      threshold=threshold)
    
    # visualising the part of the code which is not adversarial
    fun.view_embeddings(code1[:,8:,:,:])

    print(
    """
    Checking performance on ground truth change maps
    We output the code subtraction with the model and on the baseline (simple
    rasters subtraction)
    """)
    
    # getting confusion matrix on 
    # making a list of possible thresholds for the confusion matrix
    thresholds = [0, 0.46, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.25, 2.5, 2.75, 3]
    
    ## evaluating the model
    pred, y, classes = eval_model.evaluate_model(gt_change, trained_model,
                                                 args)

    # calculating the confusion matrix
    for thresh in thresholds:
        
        # converting to binary
        binary_vec = fun.convert_binary(pred, thresh)
        
        # visualizing the confusion matrix
        fun_metrics.confusion_matrix_visualize(binary_vec, y, thresh)
        
        # evaluating the precision per class
        fun_metrics.class_precision(binary_vec, y, classes)
        
    # ROC
    fun_metrics.visualize_roc(y, pred, return_thresh=False)
    
    ## evaluate the baseline
    # get prediction and targets with the baseline
    pred_alt, pred_rad, y = eval_model.evaluate_baseline(gt_change)
    
    ## making the ROC curve
    thresh = fun_metrics.visualize_roc(y, pred_alt, return_thresh=True)
    fun_metrics.visualize_roc(y, pred_rad)
    
    # calculating the confusion matrix for alt
    for thresh in thresholds:
        
        # converting to binary
        binary_vec_alt = fun.convert_binary(pred_alt, thresh)
        
        # visualizing the confusion matrix
        fun_metrics.confusion_matrix_visualize(binary_vec_alt, y, thresh)
        
        # evaluating the precision per class
        fun_metrics.class_precision(binary_vec_alt, y, classes)
    
    # calculating the confusion matrix for radiometry
    for thresh in thresholds:
        
        # converting to binary
        binary_vec_rad = fun.convert_binary(pred_rad, thresh)
        
        # visualizing the confusion matrix
        fun_metrics.confusion_matrix_visualize(binary_vec_rad, y, thresh)

    
    print(
    """
    Visualizing result for the ground truth
    """)
    
    for i in range(10,20):
        # loading the raster
        nb = i
        rast1 = gt_change["1954"][nb][None,1:,:,:]
        rast2 = gt_change["1970"][nb][None,1:,:,:]
        
        # loading the gt
        gts = [gt_change["1954"][nb][None,0,:,:].squeeze(), 
               gt_change["1970"][nb][None,0,:,:].squeeze()]
        
        
        cmap, dccode, code1, code2 = fun.change_detection(rast1, rast2, trained_model,
                                                          args,
                                                          visualization=True,
                                                          threshold=threshold, gts=gts)
        
    print(
    """
    Performing normalized mutual information for continuous variables
    """)
        
    ## extracting the codes
    # load list of codes
    list_codes = []
    
    # convert the rasters into codes
    for year in gt_change:
        
        if args.split:
            list_codes += [trained_model.encoder(fun.torch_raster(rast[None,1:,:,:]))[:,:8,:,:] for rast in gt_change[year]]
        else:
            list_codes += [trained_model.encoder(fun.torch_raster(rast[None,1:,:,:])) for rast in gt_change[year]]
        
    # convert them back to numpy matrixes
    np_codes = [rast.detach().cpu().numpy() for rast in list_codes]
        
    # stacking into one matrix
    matrix_codes = np.stack(np_codes, axis=0)
    matrix_codes = matrix_codes.squeeze()
    
    # reshaping
    if args.split:
        flat_codes = matrix_codes.transpose(0,2,3,1).reshape((matrix_codes.shape[0]*32*32, 8))
    else:
        flat_codes = matrix_codes.transpose(0,2,3,1).reshape((matrix_codes.shape[0]*32*32, 16))
        
    ## extracting the altitude
    # load list of mns
    list_mns = []
    
    # loading the mns
    for year in gt_change:
        list_mns += [rast[1,:,:] for rast in gt_change[year]]
        
    # reshape to have one single matrix
    flat_mns = fun.prepare_nmi(list_mns)
    
    ## extracting the radiometry
    # load list of rad
    list_rad = []
    
    # loading the rad
    for year in gt_change:
        list_rad += [rast[2,:,:] for rast in gt_change[year]]
        
    # reshape to have one single matrix
    flat_rad = fun.prepare_nmi(list_rad)
    
    ## extracting the labels
    # load list of labels
    list_labels = []
    
    # loading the labels
    for year in gt_change:
        list_labels += [rast[0,:,:] for rast in gt_change[year]]
        
    # transposing into one matrix
    flat_labels = fun.prepare_nmi(list_labels, discrete=True)
    
    ## removing the no data values
    # getting the nodata matrix
    data_index = flat_labels != 0
    
    # applying the mask
    labels_clean = flat_labels[data_index]
    codes_clean = flat_codes[data_index, :]
    mns_clean = flat_mns[data_index]
    rad_clean = flat_rad[data_index]
    
    ## getting the number of pixels per classes
    nb_build = np.count_nonzero(labels_clean == 1)
    nb_road = np.count_nonzero(labels_clean == 2)
    nb_field = np.count_nonzero(labels_clean == 3)
    nb_classes = (nb_build, nb_road, nb_field)
    
    ## spliting the dataset according to the class
    # loading the data
    buildings_idx = labels_clean == 1
    roads_idx = labels_clean == 2
    fields_idx =  labels_clean == 3
    
    # putting into a list
    classes_idx = [buildings_idx, roads_idx, fields_idx]
    
    # calculating the NMI for the codes
    fun_metrics.NMI_continuous_discrete(labels_clean, codes_clean,
                                        nb_classes, [1,2,3], classes_idx)
    # calculating the NMI for the mns
    fun_metrics.NMI_continuous_discrete(labels_clean, mns_clean[:,None],
                                        nb_classes, [1,2,3], classes_idx)
    
    # calculating the NMI for the rad
    fun_metrics.NMI_continuous_discrete(labels_clean, rad_clean[:,None],
                                        nb_classes, [1,2,3], classes_idx)
    
    print(
    """
    Making a linear SVM
    """)
    
    ## linear svm with the model
    conf_mat_model, class_report_model = fun_metrics.svm_accuracy_estimation(codes_clean,
                                                                             labels_clean)
    
    ## linear svm with the mns
    conf_mat_mns, class_report_mns = fun_metrics.svm_accuracy_estimation(mns_clean,
                                                                             labels_clean)
    
    ## linear svm with the rad
    conf_mat_rad, class_report_rad = fun_metrics.svm_accuracy_estimation(rad_clean,
                                                                             labels_clean)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    