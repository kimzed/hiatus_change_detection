# Project hiatus
# main script
# 12/10/2020
# Cédric BARON

# loading required packages
import numpy as np
import torch
import mock
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import random
from torch.utils.tensorboard import SummaryWriter
from numpy import load
from rasterio.plot import show
import os

import imageio

os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

# importing our functions
import utils as fun
import model as mod


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
        if year in file and "gt" not in file:
            s_rasters_clipped[year].append(load(file))
            
# loading the ground truth (corresponding dates)
for file in list_files:
    if "gt" in file:
        gt += [load(file)]
        
## loading the GT rasters
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection/data/GT_np")

# dict to store our GT rasters
gt_change ={"1966":[], "1970":[], "1978":[], "1989":[]}

# getting the list of the files
list_files = os.listdir()
list_files.sort()

# loading the matrixes in the dict per year
for file in list_files:
    for year in gt_change:
        if year in file:
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

## loading the model in case we have it already
load_model = False

## working with tensorboard
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")
writer = SummaryWriter('runs/1611_1_testcmap')

if load_model == True:
    args = mock.Mock() 
    args.n_channel = 1
    args.conv_width = [8,4,8,8,16,8]
    args.dconv_width = [8,4,8,4]
    trained_model = mod.SegNet(args.n_channel, args.conv_width, args.dconv_width)
    trained_model.load_state_dict(torch.load("models/AE_D_model"))
    trained_model.eval()

# otherwise, train it
else:
    #stores the parameters
    args = mock.Mock() 
    args.n_epoch = 100
    args.batch_size = 64
    args.n_channel = 1
    args.conv_width = [8,8,16,16,16,16]
    args.dconv_width = [8,8,8,8]
    args.cuda = 1
    args.lr = 0.002
    trained_model, trained_discr = mod.train_full(args, train_data, writer)

## saving the model
save_model = False

# saving the model
if save_model == True:
    torch.save(trained_model.state_dict(), "models/pixel_wise_100")

## visualizing the result
for i in range(5):
    
    # visualizing training raster
    raster = datasets["train"][i]
    fun.visualize(raster, third_dim=False)
    
    # visualizing prediction
    pred = trained_model.predict(raster[None,:,:,:].float().cuda())[0].cpu()
    fun.visualize(pred.detach().numpy().squeeze(), third_dim=False)
    
### view the embeddings in the model
'''

Now we are going to visualize various embeddings in the model itself

'''


# visualizing for a random index number the inner embeddings
fun.view_u(datasets["train"], trained_model, random.randint(0, 900))

# visualizing embedding inside the model
nb = random.randint(0, 900)
fun.view_u(s_rasters_clipped["1966"], trained_model, nb)
fun.view_u(s_rasters_clipped["1970"], trained_model, nb)

"""

Performing change detection analysis on manually modified data

"""

### working on data manipulation to test the model
imageio.imwrite('house.jpg', s_rasters_clipped["1966"][12][1,:,:])
imageio.imwrite('field.jpg', s_rasters_clipped["1966"][14][1,:,:])
imageio.imwrite('house_alt.jpg', s_rasters_clipped["1966"][12][0,:,:])
imageio.imwrite('field_alt.jpg', s_rasters_clipped["1966"][14][0,:,:])
    
# importing modified image
os.chdir("/home/adminlocal/Bureau/project_hiatus/data")  
field = imageio.imread('raster_modified/field_house.jpg')
field.tofile('field_house.raw') # Create raw file
field_house_raw = np.fromfile('field_house.raw', dtype=np.uint8)
field_house_raw.shape
field_house_raw.shape = (128,128)

field = imageio.imread('raster_modified/field.jpg')
field.tofile('field.raw') # Create raw file
field_raw = np.fromfile('field.raw', dtype=np.uint8)
field_raw.shape
field_raw.shape = (128,128)


field = imageio.imread('raster_modified/field_house_alt.jpg')
field.tofile('field_house_alt.raw') # Create raw file
field_house_alt = np.fromfile('field_house_alt.raw', dtype=np.uint8)
field_house_alt.shape
field_house_alt.shape = (128,128)

field = imageio.imread('raster_modified/field_alt.jpg')
field.tofile('field_alt.raw') # Create raw file
field_alt = np.fromfile('field_alt.raw', dtype=np.uint8)
field_alt.shape
field_alt.shape = (128,128)

field_raw = np.stack((field_alt, field_raw), axis=0)
field_house_raw = np.stack((field_house_alt, field_house_raw), axis=0)
    
## running cd model
rast1 = field_raw[None,:,:,:]
rast2 = field_house_raw[None,:,:,:]
threshold = 3

# computing change raster
dccode, code1, code2, cmap = fun.change_detection(rast1, rast2, trained_model, threshold)

# visualizing result
fun.visualize(rast1[:,:,:].squeeze(), third_dim = False)
fun.visualize(rast2[:,:,:].squeeze(), third_dim = False)
fun.view_embeddings(dccode)

"""

Performing change detection analysis on actual data

"""
ind = random.randint(0, 1500)

## running cd model
rast1 = s_rasters_clipped["1954"][144][None,:,:,:]
rast2 = s_rasters_clipped["1966"][144][None,:,:,:]

ind = random.randint(0, 900)

fun.visualize(s_rasters_clipped["1954"][482][:,:,:], third_dim=False)
fun.visualize(s_rasters_clipped["1966"][482][:,:,:], third_dim=False)

    
threshold = 2.5

# computing change raster
dccode, code1, code2, cmap = fun.change_detection(rast1, rast2, trained_model, threshold)

# visualizing result
fun.visualize(rast1[:,:,:].squeeze(), third_dim = False)
fun.visualize(rast2[:,:,:].squeeze(), third_dim = False)
fun.view_embeddings(dccode)

# visualize the binary change detection raster
show(cmap)


"""

Checking performance on ground truth change maps

"""


# creating a list with all the scores per year
acc_list = {"1966":[], "1970":[], "1978":[], "1989":[]}

# making a list of possible thresholds
thresholds = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 8, 9, 10, 11]

for year in gt_change:
    
    # loading list of rasters per year
    list_gt_rasts = gt_change[year]
    
    for thresh in thresholds:
        
        # computinf accuracy on the list of rasters
        acc_list[year].append(fun.AccuracyModel(list_gt_rasts, trained_model, thresh))
        
