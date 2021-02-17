# Project hiatus
# main script with a parser for the model
# 12/10/2020
# Cédric BARON

# loading required packages
import os
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


import warnings
warnings.filterwarnings('ignore')

def main():
    
    # create the parser with all arguments
    parser = argparse.ArgumentParser(description='Auto-encoder Time Adversarial Model')
    
    # Optimization arguments
    parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='Multiplicative factor used on learning rate at `lr_steps`')
    parser.add_argument('--lr_steps', default=[50, 100, 1000, 1500], help='List of epochs where the learning rate is decreased by `lr_decay`')
    parser.add_argument('--epochs', default=1, type=int, help='Number of epochs to train. If <=0, only testing will be done.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
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
    parser.add_argument('--data_fusion', default=1, help='Including data fusion')
    parser.add_argument('--rad_input', default=1, help='In case of no data_fusion, we indicate which input we want')
    parser.add_argument('--adversarial', default=0, help='Making the model adversarial')
    parser.add_argument('--defiance', default=0, help='Including defiance')
    parser.add_argument('--split', default=0, help='Making a split on the code')
    parser.add_argument('--auto_encod', default=1, help='Activating the auto-encoder')
    parser.add_argument('--name_model', default="AE_Mmodal", help='Name of the file to save the model')
    parser.add_argument('--output_dir', default="evaluation_models/", help='Name of the dir to save the model')
    
    # Encoder
    parser.add_argument('--conv_width', default=[8,8,16,16,16], help='Layers size')
    
    # Decoder
    parser.add_argument('--dconv_width', default=[16,16,8,8,8], help='Layers size')
    
    # Defiance
    parser.add_argument('--def_width', default=[16,16,16,16,16], help='Layers size')
    
    # Discriminator
    parser.add_argument('--nb_channels_split', default=16, type=int, help='Number of channels for the input to the discriminator')
    parser.add_argument('--disc_width', default=[16,16,16,16,16,16,16,16,16], help='Layers size')
    parser.add_argument('--disc_loss_weight', default=0.1, type=float, help='Weight applied on the adversarial loss with full model')
    parser.add_argument('--opti_adversarial_encoder', default=0, help='Trains the encoder weights')
    
    args = parser.parse_args()
    args.start_epoch = 0
    
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
    train_idx, val_idx = train_test_split(list(range(len(gt_change["1970"]))), test_size=0.20, random_state=1)
    
    # we load the train and test sets for GT
    for year in gt_change:
        gt_change_test[year] = Subset(gt_change[year], val_idx)
        
    for year in gt_change:
        gt_change[year] = Subset(gt_change[year], train_idx)
        
    # training the model
    trained_model = train.train_full(args, train_data, gt_change_test)
    
    return args, gt_change, numpy_rasters, trained_model, train_data

###############################################################################
###############################################################################
###############################################################################
        
if __name__ == "__main__":
    
    print(
    """
    Training the model
    """)
    
    # running the model
    args, gt_change, numpy_rasters, trained_model, datasets = main()
    

    print(
    """
    We now test the results for several models
    """)
    
    # boolean to allow evaluation or not
    evaluation = False
    
    # performing evaluation on the different models
    if evaluation:
        print("AE_rad")
        eval_model.evaluate_model("AE_rad", gt_change)
        print("AE_rad+DAN")
        eval_model.evaluate_model("AE_rad+DAN", gt_change)
        print("AE_Mmodal")
        eval_model.evaluate_model("AE_Mmodal", gt_change)
        print("AE_Mmodal+DAN")
        eval_model.evaluate_model("AE_Mmodal+DAN", gt_change)
        print("AE_Mmodal+DAN+split")
        eval_model.evaluate_model("AE_Mmodal+DAN+split", gt_change)
        print("AE_alt+DAN")
        eval_model.evaluate_model("AE_alt+DAN", gt_change)
        print("bayesian_model")
        eval_model.evaluate_model("bayesian_model", gt_change)

    
    
    
    
    