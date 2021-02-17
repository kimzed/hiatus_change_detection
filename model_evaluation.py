# Project hiatus
# script used to evaluate our models and analyse the results
# 16/11/2020
# Cédric BARON

# loading required packages
import os
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch
from sklearn.linear_model import LinearRegression
import sklearn
import random

# for manual visualisation
from rasterio.plot import show

# importing our functions
import utils as fun
import train as train
import evaluate as eval_model
import metrics as fun_metrics


# putting the right work directory
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

import warnings
warnings.filterwarnings('ignore')



print(
"""

Loading the model and the data

""")

# loading the dataset, getting a raster for later data visualisation
# after every epoch
import frejus_dataset
# loading the data
train_data, gt_change, numpy_rasters = frejus_dataset.get_datasets(["1954","1966","1970", "1978", "1989"])

## loading the model
name_model = "AE_Mmodal+DAN+split"
dict_model = torch.load("evaluation_models/"+name_model)
args = dict_model["args"]
trained_model = fun.load_model_from_dict(dict_model)

# setting the seed
fun.set_seed(args.seed, args.cuda)

## we take a test set of the gt_change for evaluation (20%)
# creating a new dict for gt test
gt_change_test = {}
# getting a single subset list throughout the years
train_idx, val_idx = train_test_split(list(range(len(gt_change["1970"]))), test_size=0.20, random_state=1)

# loading the GT
for year in gt_change:
    gt_change[year] = Subset(gt_change[year], train_idx)
    

print(
"""
Checking performance on ground truth change maps
We output the code subtraction with the model and on the baseline (simple
rasters subtraction)
""")

## generating prediction for the model
pred, y, classes = eval_model.generate_prediction_model(gt_change, trained_model,
                                             args)


## evaluate the baseline
# get prediction and targets with the baseline
pred_alt, pred_rad, y = eval_model.generate_prediction_baseline(gt_change)

## making the ROC curve
threshold=fun_metrics.visualize_roc(y, pred_alt, return_thresh=True)
fun_metrics.iou_accuracy(pred_alt, threshold, y, classes)
threshold=fun_metrics.visualize_roc(y, pred_rad, return_thresh=True)
fun_metrics.iou_accuracy(pred_rad, threshold, y, classes)

# ROC for the model
threshold=fun_metrics.visualize_roc(y, pred, return_thresh = True)

## getting the IUC and the accuracy
fun_metrics.iou_accuracy(pred, threshold, y, classes)

print(
"""
Visualizing change detection on the ground truth
""")

for i in range(30,35):
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

# load the data and the baselines
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

# calculating the NMI for the both inputs
dem_rad = np.concatenate((rad_clean[:,None], mns_clean[:,None]), axis=1)
fun_metrics.NMI_continuous_discrete(labels_clean, dem_rad,
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

# loading two dictionary for cross-validation
gt_change_train = {}
gt_change_test = {}

# creating test and train data on distinct locations
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
    if year == "1970":
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


print("""
   Now we do transfer learning    (bayesian model)
""")

## loading the pre trained model
dict_model = torch.load("evaluation_models/test_transfer_aleo")

dict_model["args"].epochs = 1
dict_model["args"].defiance = 1
dict_model["args"].save = 0
dict_model["args"].load_best_model = 1
dict_model["args"].grad_clip = 0
dict_model["args"].name_model = "bayesian_model"

# updating the args
args = dict_model["args"]

# starting the run
trained_model = train.train_full(args, train_data, gt_change, dict_model)

print("""
      Performing change detection with the alternative model (training the model
      and then assessing the result)
""")

# list of years
years = ["1954","1966", "1970", "1978", "1989"]

# loading the data
import frejus_dataset
train_data, gt_change, numpy_rasters = frejus_dataset.get_datasets(["1954","1966","1970", "1978", "1989"])

# loading the args of the pre-trained model
dict_model = torch.load("evaluation_models/pre_trained_baseline")
args = dict_model["args"]
args.rad_input=1

# setting the number of epochs
args.epochs = 20

# getting th year for the first rasters
for year1 in years:
    
    # getting the year for the second raster
    for year2 in years:
        
        # checking that both year are not the same year
        if year1 != year2 and year2 > year1:
            
            # naming the model
            args.name_model = year1+"to"+year2+"_baseline"
            
            # loading the data
            train_data, _, numpy_rasters = frejus_dataset.get_datasets([year1,year2])
            
            # taking two years and converting into torch
            numpy_rasters[year1] = [fun.torch_raster(raster, cuda=False) for raster in numpy_rasters[year1]]
            numpy_rasters[year2] = [fun.torch_raster(raster, cuda=False) for raster in numpy_rasters[year2]]
            
            
            # training the model
            trained_model = train.train_full_alternative_model(args, numpy_rasters, dict_model)
            
    
## evaluating the model
pred, y, classes = eval_model.generate_prediction_baseline_model(gt_change, args)

# ROC
threshold=fun_metrics.visualize_roc(y, pred, return_thresh=False)

# accuracy and IoU
fun_metrics.iou_accuracy(pred, 0.69, y, classes)

print("""
      Visualizing change detection on the ground truth
      """)


for i in range(10):
    # loading the raster
    nb = i
    rast1 = gt_change["1954"][nb][None,1:,:,:]
    rast2 = gt_change["1970"][nb][None,1:,:,:]
    
    # loading the gt
    gts = [gt_change["1954"][nb][None,0,:,:].squeeze(), 
           gt_change["1970"][nb][None,0,:,:].squeeze()]
    
    
    fun.change_detection_baseline(rast1, rast2, ["1954", "1970"], args,
                                                      visualization=True,
                                                      threshold=1.3, gts=gts)
    
    
print("""
      
      Estimating correlation between codes, DEM and rad
      
      """)

# getting the index for cross-validation
train_idx, val_idx = train_test_split(list(range(len(gt_change["1954"]))), test_size=0.25)

# empty dicts to store train and test sets
gt_change_train = {}
gt_change_test = {}

# loading train and test sets
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

# training the model for mns
lr_mns = LinearRegression()
lr_mns.fit(codes_train, mns_train)      
pred_mns = lr_mns.predict(codes_test)  
mae_mns = sum(abs(pred_mns - mns_test)) / mns_test.shape[0]
r2_mns = sklearn.metrics.r2_score(mns_test, pred_mns)

#print(mae_mns)
print("R2 for mns is %1.2f" % (r2_mns))
print("\n")
print(abs(lr_mns.coef_).mean())

# training the model for rad
lr_rad = LinearRegression()
lr_rad.fit(codes_train, rad_train)      
pred_rad = lr_rad.predict(codes_test)  
mae_rad = sum(abs(pred_rad - rad_test)) / mns_test.shape[0]
r2_rad = sklearn.metrics.r2_score(rad_test, pred_rad)

#print(mae_rad)
print("R2 for rad is %1.2f" % (r2_rad))
print("\n")
print(abs(lr_rad.coef_).mean())
    
### computing the MI
# adding test data to train data
codes_train = np.concatenate((codes_train, codes_test), axis=0)
mns_train = np.concatenate((mns_train, mns_test), axis=None)
rad_train = np.concatenate((rad_train, rad_test), axis=None)

## binning the data
# getting the value of the quantiles
values_dem_cut = np.quantile(mns_train, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
values_rad_cut = np.quantile(rad_train, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

# binning the data with the quantiles
mns_discrete = np.digitize(mns_train,bins=values_dem_cut)
rad_discrete = np.digitize(rad_train,bins=values_rad_cut)

# lists to store class related indexes
classes_dem_idx = []
classes_rad_idx = []

# lists to store the number of samples per class
nb_classes_dem = []
nb_classes_rad = []

for i in range(10):
    
    ## class related data for DEM
    # boolean per class
    class_idx = mns_discrete == i
    classes_dem_idx.append(class_idx)
    # number of sample of the class
    nb_classes_dem.append(np.count_nonzero(mns_discrete == i))
    
    # same opertation, for the radiometry
    class_idx = rad_discrete == i
    classes_rad_idx.append(class_idx)
    nb_classes_rad.append(np.count_nonzero(rad_discrete == i))
    
    

# calculating the NMI for DEM
mi_mns = fun_metrics.NMI_continuous_discrete(mns_discrete, codes_train,
                                    nb_classes_dem, list(range(10)), classes_dem_idx)
print("%1.2f" % (mi_mns))

# calculating the NMI for rad
mi_rad = fun_metrics.NMI_continuous_discrete(rad_discrete, codes_train,
                                    nb_classes_rad, list(range(10)), classes_rad_idx)
print("%1.2f" % (mi_rad))


print("""
      
      calculating the MI per raster
      
      """)

# getting a random raster from the GT
nb = random.randint(0, 40)
raster = gt_change["1970"][nb]

# getting the MI per raster
print("rad")
fun.MI_raster(raster, "AE_rad")
print("\n")
print("Mmodal")
fun.MI_raster(raster, "AE_Mmodal", visu=True)
print("\n")
print("DAN")
fun.MI_raster(raster, "AE_Mmodal+DAN")
print("\n")

print("""
      
Doing tsne visualization on the ground truth
      
""")

# tsne on a single raster with different models
nb = random.randint(0, 40)
raster = gt_change["1970"][nb]
fun.tsne_visualization(raster, trained_model, "AE_rad")
fun.tsne_visualization(raster, trained_model, "AE_rad+DAN")
fun.tsne_visualization(raster, trained_model, "AE_Mmodal")
fun.tsne_visualization(raster, trained_model, "AE_Mmodal+DAN")

# tsne on the whole dataset with different model
fun.tsne_dataset(gt_change, "AE_rad")
fun.tsne_dataset(gt_change, "AE_rad+DAN")
fun.tsne_dataset(gt_change, "AE_Mmodal")
fun.tsne_dataset(gt_change, "AE_Mmodal+DAN")
fun.tsne_dataset(gt_change, "AE_Mmodal+DAN+split")


print(
"""
We now test the results for several models
""")

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


print(
    """
    Visualizing some predictions for the autoencoder
    """)

# removing the year vectors
datasets = [raster[0] for raster in train_data]

for i in range(10,15):
    
    # visualizing training raster
    raster = datasets[i]
    fun.visualize(raster, third_dim=False)
    
    # visualizing prediction
    pred = trained_model.predict(raster[None,:,:,:].float().cuda(), args)[0].cpu()
    pred = fun.numpy_raster(pred)
    fun.visualize(pred, third_dim=False, defiance=args.defiance)
    
    # scatter plot for the defiance
    if args.defiance:
        fun.scatter_aleo(raster[1,:,:], pred[1,:,:], pred[2,:,:])
        
print(
    '''
    Now we are going to visualize various embeddings in the model itself
    ''')

# visualizing for a random index number the inner embeddings
fun.view_u(datasets, trained_model, args, random.randint(0, 900))

# visualizing embedding inside the model
nb = random.randint(0, 900)
print(nb)
fun.view_u(numpy_rasters["1966"], trained_model, args, nb)
fun.view_u(numpy_rasters["1970"], trained_model, args, nb)


print(
    """
    Performing change detection analysis on some examples
    """)


# loading two random rasters
nb = random.randint(0, 900)
print(i)
rast1 = numpy_rasters["1954"][i][None,:,:,:]
rast2 = numpy_rasters["1989"][i][None,:,:,:]

# computing change raster
cmap, dccode, code1, code2 = fun.change_detection(rast1, rast2, trained_model, args,
                                                  threshold=threshold, visualization=True)



    