# Project hiatus
# main script
# 12/10/2020
# Cédric BARON

# loading required packages
import torch
import mock
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import random
from torch.utils.tensorboard import SummaryWriter
from numpy import load
from rasterio.plot import show
import os
import numpy as np


os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

# importing our functions
import utils as fun
import model as mod
import train as train
import evaluate as eval_model
import metrics as fun_metrics

"""

Loading the existing rasters

"""

# changing working directory
os.chdir("data/np_data")

# list rasters
list_files = os.listdir()

# sorting the names to have similar order
list_files.sort(reverse=True)

# storing our rasters per year in a dictionary
s_rasters_clipped = {"1954":[], "1966":[], "1970":[], "1978":[], "1989":[]}

# loading the list for the ground truth
gt = []

# loading the rasters
for year in s_rasters_clipped:
    for file in list_files:
        if file[:4] == year in file and "gt" not in file:
            s_rasters_clipped[year].append(load(file))
            
# loading the ground truth (corresponding dates)
for year in s_rasters_clipped:
    for file in list_files:
        if file[:4] == year in file and "gt" in file:
            gt += [load(file)]
        
## loading the GT rasters
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection/data/GT_np")

# dict to store our GT rasters
gt_change = {"1954":[], "1966":[], "1970":[], "1978":[], "1989":[]}

# getting the list of the files
list_files_gt = os.listdir()
list_files_gt.sort()

# loading the matrixes in the dict per year
for file in list_files_gt:
    for year in gt_change:
        if year in file and "class" in file:
            gt_change[year].append(load(file))
      
"""

we now build our dataset as a list of tensors

"""

# stacking up the  samples into a list
m_samples = []

for year in s_rasters_clipped:
    
    m_samples += s_rasters_clipped[year]

def train_val_dataset(dataset, gt, val_split=0.25):
    """
    param: list of rasters as numpy objects and percentage of test data
    fun: outputs a dictionary with training and test data
    """
    
    # getting ids for training and validation sets
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    
    # subsetting into training and validation, storing into a dictionary
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    
    # subsetting the groundtruth for the adversarial part
    datasets['gt_train'] = Subset(gt, train_idx)
    datasets['gt_val'] = Subset(gt, val_idx)
    
    return datasets

# loading the torch data without batch
datasets = train_val_dataset(m_samples, gt)

# extracting evals, converting into pytorch tensors
datasets["val"] = [torch.from_numpy(obs) for obs in datasets["val"]]
datasets["gt_val"] = [torch.from_numpy(obs) for obs in datasets["gt_val"]]

# extracting only images for the training, converting into tensors
datasets["train"] = [torch.from_numpy(obs) for obs in datasets["train"]]
datasets["gt_train"] = [torch.from_numpy(obs) for obs in datasets["gt_train"]]

# merging val and train because we want more samples
datasets["train"] = datasets["train"] + datasets["val"]
datasets["gt_train"] = datasets["gt_train"] + datasets["gt_val"]

## we need to combine images and labels for the discriminator
train_data = []

for i in range(len(datasets["train"])):
   train_data.append([datasets["train"][i], datasets["gt_train"][i]])

# adding number of sample
n_train = len(datasets["train"])

"""

we build and train the auto-encoder model
the model is made in the functions file

"""

# setting parameters of the model
data_fusion = True
adversarial = True
defiance = False
split = True

## loading the model in case we have it already
load_model = False

## working with tensorboard
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")
writer = SummaryWriter('runs/0212_1_test')

# loading a raster to check the models updates
ex_raster = torch.from_numpy(s_rasters_clipped["1970"][715][None,:,:,:]).cuda().float()

if load_model:
    args = mock.Mock() 
    args.n_channel = 1
    args.conv_width = [8,8,16,16,16,16]
    args.dconv_width = [8,8,8,8]
    trained_model.encoder = mod.Encoder(args.n_channel, args.conv_width)
    trained_model.encoder.load_state_dict(torch.load("models/0212_advautoenc_20epoch"))
    trained_model.encoder.eval()

# otherwise, train it
else:
    #stores the parameters
    args = mock.Mock()
    args.n_epoch = 20
    args.batch_size = 92
    args.n_channel = 1
    args.conv_width = [8,8,16,16,16,16]
    args.dconv_width = [8,8,8,8]
    args.cuda = 1
    args.lr_steps = [50, 70, 90]
    args.lr_decay = 0.5
    args.lr = 0.05
    trained_model = train.train_full(args, train_data, writer, gt_change,
                                                    ex_raster,
                                                    data_fusion=data_fusion,
                                                    adver=adversarial,
                                                    defiance=defiance,
                                                    split=split)
    # getting into eval() mode
    trained_model.encoder.eval()
    trained_model.decoder.eval()

## saving the model
save_model = False

# saving the model
if save_model:
    torch.save(trained_model.encoder.state_dict(), "models/0312_test")

## visualizing the result
for i in range(5):
    
    # visualizing training raster
    raster = datasets["train"][i]
    fun.visualize(raster, third_dim=False)
    
    # visualizing prediction
    pred = trained_model.predict(raster[None,:,:,:].float().cuda(), data_fusion=data_fusion)[0].cpu()
    fun.visualize(pred.detach().numpy().squeeze(), third_dim=False, defiance=defiance)

'''

Now we are going to visualize various embeddings in the model itself

'''

# visualizing for a random index number the inner embeddings
fun.view_u(datasets["train"], trained_model, random.randint(0, 900),
           data_fusion=data_fusion, split=split)

# visualizing embedding inside the model
nb = random.randint(0, 900)
fun.view_u(s_rasters_clipped["1966"], trained_model, nb, data_fusion=data_fusion,
           split=split)
fun.view_u(s_rasters_clipped["1970"], trained_model, nb, data_fusion=data_fusion,
           split=split)

"""

Checking for which year the discriminator can't predict the year

"""

count = 0

trained_model.encoder.eval()
trained_model.discr.eval()

labs = []

for i in range(900):
    
    # loading gt
    gt_rast = gt[i]
    gt_rast = torch.from_numpy(gt_rast)
    _, pred_max_gt = gt_rast.max(dim=0)
    
    # loading raster
    raster = s_rasters_clipped["1954"][i]
    
    # visualizing prediction
    code = trained_model.encoder.forward(torch.from_numpy(raster[None,:,:,:]).float().cuda())
    label = trained_model.discr.forward(code)
    
    _, pred_max = label.max(dim=1)
    
    matches = pred_max.cpu() == pred_max_gt
    
    if not matches:
        
        #print(i)
        #if label[0,1] < 0.99:
            
            # visualizing training raster
            #fun.visualize(raster, third_dim=False)
        
        count += 1   
        labs.append(label)
        


"""

Performing change detection analysis on actual data

"""

ind = random.randint(0, 900)

# interesting nbs 54-70: 783, 439,746, 201 706 715
# no change 66-70: 245 799 406 437, 715

ind = random.randint(0, 900)
print(ind)
fun.visualize(s_rasters_clipped["1966"][ind][:,:,:], third_dim=False)
fun.visualize(s_rasters_clipped["1970"][ind][:,:,:], third_dim=False)

nb = 715
ind = nb
## running cd model
rast1 = s_rasters_clipped["1966"][nb][None,:,:,:]
rast2 = s_rasters_clipped["1970"][nb][None,:,:,:]

threshold = 1

# computing change raster
cmap, dccode, code1, code2 = fun.change_detection(rast1, rast2, trained_model,
                                                  visualization=True, data_fusion=True,
                                                  threshold=threshold, split=split)

# visualising the part of the code which is not adversarial
fun.view_embeddings(code1[:,8:,:,:])


"""

Checking performance on ground truth change maps
We output the code subtraction with the model and on the baseline (simple
rasters subtraction)

"""

# getting confusion matrix on 
# making a list of possible thresholds for the confusion matrix
thresholds = [0, 0.46, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.25, 2.5, 2.75, 3]

## evaluating the model
pred, y, classes = eval_model.evaluate_model(gt_change, trained_model,
                                             data_fusion=data_fusion,
                                             split=split)

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

"""

Visualizing result for the ground truth

"""

for i in range(2,3):
    # loading the raster
    nb = i
    rast1 = gt_change["1954"][nb][None,1:,:,:]
    rast2 = gt_change["1970"][nb][None,1:,:,:]
    
    # loading the gt
    gts = [gt_change["1954"][nb][None,0,:,:].squeeze(), 
           gt_change["1970"][nb][None,0,:,:].squeeze()]
    
    
    cmap, dccode, code1, code2 = fun.change_detection(rast1, rast2, trained_model,
                                                      visualization=True, data_fusion=True,
                                                      threshold=threshold, gts=gts)
    
    
"""

Performing normalized mutual information for continuous variables

"""
    
## extracting the codes
# load list of codes
list_codes = []

# convert the rasters into codes
for year in gt_change:
    list_codes += [trained_model.encoder(fun.torch_raster(rast[None,1:,:,:])) for rast in gt_change[year]]

# convert them back to numpy matrixes
np_codes = [rast.detach().cpu().numpy() for rast in list_codes]
    
# stacking into one matrix
matrix_codes = np.stack(np_codes, axis=0)
matrix_codes = matrix_codes.squeeze()

# reshaping
flat_codes = matrix_codes.transpose(0,2,3,1).reshape((155*32*32, 16))

## extracting the labels
# load list of labels
list_labels = []

# loading the labels
for year in gt_change:
    
    # reshaping and loading in the list
    for rast in gt_change[year]:
        
        # loading the labels
        rast_labels = rast[0,:,:]
        
        # reshaping
        rast_resh =  fun.regrid(rast_labels.reshape(rast.shape[1:]), 32, 32, "nearest")
        rast_resh = np.rint(rast_resh)
        
        # storing into our list
        list_labels.append(rast_resh)


# stacking into one matrix
matrix_labels = np.stack(list_labels, axis=0)

# reshaping
flat_labels = matrix_labels.reshape((155*32*32))

## removing the no data values
# getting the nodata matrix
data_index = flat_labels != 0

# loading codes and labels with mask
labels_clean = flat_labels[data_index]
codes_clean = flat_codes[data_index, :]

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

# calculating the NMI
fun_metrics.NMI_continuous_discrete(labels_clean, codes_clean,
                                    nb_classes, [1,2,3], classes_idx)


