# Project hiatus
# main script with a parser for the model
# 12/10/2020
# Cédric BARON

# loading required packages
import os
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# for manual visualisation
from rasterio.plot import show

# putting the right work directory
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

# importing our functions
import utils as fun
import train as train
import evaluate as eval_model
import metrics as fun_metrics
import model as mod

def main():
    
    # create the parser with all arguments
    parser = argparse.ArgumentParser(description='Auto-encoder Time Adversarial Model')
    
    # Optimization arguments
    parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default=[50, 100, 1000, 1500], help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--optim', default='adam', help='Optimizer: sgd|adam')
    parser.add_argument('--grad_clip', default=0, type=float, help='Element-wise clipping of gradient. If 0, does not clip')
    
    # Learning process arguments
    parser.add_argument('--cuda', default=1, type=int, help='Bool, use cuda')
    parser.add_argument('--test_auc', default=1, type=int, help='Test each n-th epoch during training')
    parser.add_argument('--load_best_model', default=1, type=int, help='Load the model with the best result')

    # Dataset
    parser.add_argument('--dataset', default='frejus_dataset', help='Dataset name: frejus_dataset')
    
    # Model
    parser.add_argument('--seed', default=1, type=int, help='Seed for random initialisation')
    parser.add_argument('--save', default=0, type=int, help='Seed for random initialisation')
    parser.add_argument('--data_fusion', default=0, help='Including data fusion')
    parser.add_argument('--adversarial', default=0, help='Making the model adversarial')
    parser.add_argument('--defiance', default=0, help='Including defiance')
    parser.add_argument('--split', default=0, help='Making a split on the code')
    parser.add_argument('--auto_encod', default=1, help='Activating the auto-encoder')
    
    # Encoder
    parser.add_argument('--conv_width', default=[8,8,16,16,16], help='Layers size')
    
    # Decoder
    parser.add_argument('--dconv_width', default=[16,16,8,8,8], help='Layers size')
    
    # Defiance
    parser.add_argument('--def_width', default=[16,16,16,16,16], help='Layers size')
    
    # Discriminator
    parser.add_argument('--nb_channels_split', default=8, type=int, help='Number of channels for the input to the discriminator')
    parser.add_argument('--disc_width', default=[32,16,16,16,16,16,16,16,16], help='Layers size')
    parser.add_argument('--nb_trains_discr', default=1, type=int, help='Number of times the discriminator is trained compared to the autoencoder')
    parser.add_argument('--disc_loss_weight', default=0.15, type=float, help='Weight applied on the adversarial loss with full model')
    parser.add_argument('--opti_adversarial_encoder', default=0, help='Trains the encoder weights')
    
    args = parser.parse_args()
    args.start_epoch = 0
    
    # we increase the width of the encoder
    args.conv_width = [2*x for x in args.conv_width]
    
    # setting the seed
    fun.set_seed(args.seed, args.cuda)
    
    # Decide on the dataset
    if args.dataset=='frejus_dataset':
        
        # loading the dataset, getting a raster for later data visualisation
        # after every epoch
        import frejus_dataset
        # loading the data
        train_data, gt_change, numpy_rasters = frejus_dataset.get_datasets(["1954","1966","1970", "1978", "1989"])
    
    ## we take a test set of the gt_change for evaluation (20%)
    # creating a new dict for gt test
    gt_change_test = {}
    # getting a single subset list throughout the years
    train_idx, val_idx = train_test_split(list(range(len(gt_change["1970"]))), test_size=0.30)
    for year in gt_change:
        gt_change_test[year] = Subset(gt_change[year], val_idx)

    # training the model
    trained_model = train.train_full(args, train_data, gt_change_test)
    
    return args, gt_change, numpy_rasters, trained_model, train_data

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
    
# =============================================================================
#     # removing the year vector from the data
#     datasets = [data[0] for data in datasets]
# =============================================================================
    
if __name__ == "__other__": 
    
    load_model=False
    
    if load_model == True:
        trained_model, args = fun.load_model("evaluation_models/AE-MModal+DAN", "evaluation_models/AE-MModal+DAN.txt")
        
    print(
    """
    Checking performance on ground truth change maps
    We output the code subtraction with the model and on the baseline (simple
    rasters subtraction)
    """)
    
    ## evaluating the model
    pred, y, classes = eval_model.generate_prediction_model(gt_change, trained_model,
                                                 args)

    # ROC
    fun_metrics.visualize_roc(y, pred, return_thresh=False)
    
    ## evaluate the baseline
    # get prediction and targets with the baseline
    pred_alt, pred_rad, y = eval_model.generate_prediction_baseline(gt_change)
    
    ## making the ROC curve
    fun_metrics.visualize_roc(y, pred_alt, return_thresh=True)
    fun_metrics.visualize_roc(y, pred_rad)
    
    print(
    """
    Performing normalized mutual information for continuous variables
    """)
    

    codes_clean, labels_clean = fun.prepare_codes_metrics(gt_change, args, trained_model)
    
    mns_clean = fun.prepare_data_metrics(gt_change, 1)
    rad_clean = fun.prepare_data_metrics(gt_change, 2)
    
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
    conf_mat_model, class_report_model, scores_cv = fun_metrics.svm_accuracy_estimation(codes_clean,
                                                                             labels_clean)
    
    ## linear svm with the mns
    conf_mat_mns, class_report_mns, scores_cv = fun_metrics.svm_accuracy_estimation(mns_clean,
                                                                             labels_clean)
    
    ## linear svm with the rad
    conf_mat_rad, class_report_rad, scores_cv = fun_metrics.svm_accuracy_estimation(rad_clean,
                                                                             labels_clean)
    
    ### Linear svm but distinct geographical locations
    # getting ids for training and validation sets
    train_idx, val_idx = train_test_split(list(range(len(gt_change["1954"]))), test_size=0.25)

    
    gt_change_train = {}
    gt_change_test = {}
    
    for year in gt_change:
        gt_change_train[year] = Subset(gt_change[year], train_idx)
        gt_change_test[year] = Subset(gt_change[year], val_idx)
    
    # data for train
    codes_train, labels_train = fun.prepare_codes_metrics(gt_change_train, args, trained_model)
    mns_train = fun.prepare_data_metrics(gt_change_train, 1)
    rad_train= fun.prepare_data_metrics(gt_change_train, 2)
    
    # data for test
    codes_test, labels_test = fun.prepare_codes_metrics(gt_change_test, args, trained_model)
    mns_test = fun.prepare_data_metrics(gt_change_test, 1)
    rad_test = fun.prepare_data_metrics(gt_change_test, 2)
    
    ## linear svm with the model
    conf_mat_model, class_report_model, scores_cv_model = fun_metrics.svm_accuracy_estimation_2(codes_train, codes_test, labels_train, labels_test, cv=False)
    
    ## linear svm with the mns
    conf_mat_mns, class_report_mns, scores_cv_mns = fun_metrics.svm_accuracy_estimation_2(mns_train, mns_test, labels_train, labels_test, cv=False)
    
    ## linear svm with the rad
    conf_mat_rad, class_report_rad, scores_cv_rad = fun_metrics.svm_accuracy_estimation_2(rad_train, rad_test, labels_train, labels_test, cv=False)
    
    
    ## testing with only one year for train
    # getting ids for training and validation sets
    gt_change_train = {}
    gt_change_test = {}
    
    for year in gt_change:
        if year == "1989":
            gt_change_train[year] =gt_change[year]
        else:
            gt_change_test[year] = gt_change[year]
    
    # data for train
    codes_train, labels_train = fun.prepare_codes_metrics(gt_change_train, args, trained_model)
    mns_train = fun.prepare_data_metrics(gt_change_train, 1)
    rad_train= fun.prepare_data_metrics(gt_change_train, 2)
    
    # data for test
    codes_test, labels_test = fun.prepare_codes_metrics(gt_change_test, args, trained_model)
    mns_test = fun.prepare_data_metrics(gt_change_test, 1)
    rad_test = fun.prepare_data_metrics(gt_change_test, 2)
    
    ## linear svm with the model
    conf_mat_model, class_report_model, scores_cv_model = fun_metrics.svm_accuracy_estimation_2(codes_train, codes_test, labels_train, labels_test, cv=False)
    
    ## linear svm with the mns
    conf_mat_mns, class_report_mns, scores_cv_mns = fun_metrics.svm_accuracy_estimation_2(mns_train, mns_test, labels_train, labels_test, cv=False)
    
    ## linear svm with the rad
    conf_mat_rad, class_report_rad, scores_cv_rad = fun_metrics.svm_accuracy_estimation_2(rad_train, rad_test, labels_train, labels_test, cv=False)
    
    print(
    """
    We now test the results for several models
    """)
    
    import warnings
    warnings.filterwarnings('ignore')
    
    eval_model.evaluate_model("AE-RAD", gt_change)
    eval_model.evaluate_model("AE-MModal", gt_change)
    eval_model.evaluate_model("AE-MModal+DAN", gt_change)
    
    
    print("""
       Now we do transfer learning   
    """)
    
    trained_model, args = fun.load_model("evaluation_models/AE-MModal+DAN", "evaluation_models/AE-MModal+DAN.txt")

    args.epochs = 2

    train.train_full_transfer_learning(args, datasets, gt_change, trained_model)
    
        
        