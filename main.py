# Project hiatus
# main script
# 12/10/2020
# Cédric BARON

# loading required packages
import numpy as np
import rasterio
from rasterio.plot import show
import os
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
from rasterio.mask import mask
import torch
import mock
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

import imageio

os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

# importing our functions
import functions as fun

# changing working directory
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection/data/tifs")


"""

We load the bounding box

"""

# change working directory
os.chdir("/home/adminlocal/Bureau/project_hiatus/data/tifs")

## creating the bounding box for cropping our rasters
BuildingsGDF = gpd.read_file('bdtopo_bati_1954.shp')
bbox = BuildingsGDF.total_bounds
    
# setting a bounding box for cropping
glob_minx, glob_miny = bbox[0], bbox[1]
glob_maxx, glob_maxy = bbox[2], bbox[3]

# getting global boundaries in the correct format
bbox = box(glob_minx, glob_miny, glob_maxx, glob_maxy)
geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(2154))
coords = fun.getFeatures(geo)

"""

We load the rasters
Then we sample them into a grid with similar resolution

"""

# list rasters
list_files = os.listdir()
list_tifs = [name for name in list_files if name[-3:] == "tif"]
list_tifs.sort(reverse=True)

# storing our rasters per year in a dictionary
dict_rasters = {"1954":[], "1966":[], "1970":[], "1978":[], "1989":[]}

for rast_file in list_tifs:
    for year in dict_rasters:
        if year in rast_file:
            try:
                ortho = rasterio.open(rast_file)
                
                dict_rasters[year].append(ortho)
            except:
                print(rast_file)
        
    else:
        None

# creating a list for our bounding boxes
boxes = []
minx = glob_minx
miny = glob_miny

## generating the bounding boxes
while minx <= glob_maxx:
    
    # reinitializing y
    miny = glob_miny
    
    while miny <= glob_maxy:
        
        # creating bbox
        bbox = box(minx, miny, minx+64, miny+64)
        geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(2154))
        coords = fun.getFeatures(geo)
        boxes.append(coords)
        
        # updating the box boundaries
        miny += 64
    
    minx += 64


## clipping our rasters and storing them
# creating a list of lists
rasters_clipped = {}

for year in dict_rasters:
    
    # creating our year index for the adversarial part
    rasters_clipped[year] = []
    
for our_box in boxes:
    
    # list of rasters matching the box
    rasters_box = []
    
    for year in dict_rasters:
        
        for rast in dict_rasters[year]:
            try:
                # cropping the raster
                out_img, out_transform = mask(dataset=rast, all_touched=True,
                                              shapes=our_box, crop=True)
                
                # storing the raster in our list
                # removing rasters with too much zeros
                values = out_img.flatten()
                nb_zeroes = np.count_nonzero(values == 0)
                
                if nb_zeroes > len(values)/5 :
                    None
                else:
                    resh_rast = fun.regrid(out_img.reshape(out_img.shape[1:]), 128, 128)
                    rasters_box.append(resh_rast)
            
            except:
                None
    
    # we now have two * nb years rasters in the list
    # we set a condition to be sure that we have altitude and radiometry for all years
    if len(rasters_box) != len(dict_rasters)*2:
        continue
    
    else: 
        # storing the raster per year
        i = 0
        
        for year in dict_rasters:
            
            # appending the rasters into a year index
            rasters_clipped[year].append([rasters_box[i], rasters_box[i+1]])
            i += 2
    
    
"""

We now stack up the rasters into 2*128*128 dimension rasters and normalize them
using z-scores and removing minimum altitude for mns

"""

## stacking the result (every raster has two dimensions, color and altitude)
# normalizing
s_rasters_clipped = {}

for year in rasters_clipped:
    
    # stacking up along a new axis
    s_rasters_clipped[year] = [np.stack(rast) for rast in rasters_clipped[year]]
    
    # extracting alt and rad
    alt_rasts = [rast[0] for rast in rasters_clipped[year]]
    rad_rasts = [rast[1] for rast in rasters_clipped[year]]
    total_rasters_alt = np.stack(alt_rasts, axis=0)
    total_rasters_rad = np.stack(rad_rasts, axis=0)
    
    # getting stat values
    mu_alt = total_rasters_alt.mean()
    std_alt = total_rasters_alt.std()
    mu_rad = total_rasters_rad.mean()
    std_rad = total_rasters_rad.std()
    
    # normalizing
    alt_rasts = [rast - rast.min() for rast in alt_rasts]
    alt_rasts = [(rast-mu_alt)/std_alt for rast in alt_rasts]
    rad_rasts = [(rast-mu_rad)/std_rad for rast in rad_rasts]
    
    # stacking up into a dictionary
    s_rasters_clipped[year] = [np.stack((alt, rad), axis=0) for alt, rad in zip(alt_rasts, rad_rasts)]
    
# we add the year vector and stack it up
# index for the vector
i = 0

# creating the ground truth
gt = []

for year in s_rasters_clipped:
    
    # number of samples per year
    m = len(s_rasters_clipped[year])
    
    # creating the vector for the year (labels for adversarial network)
    year_vect = np.zeros(len(s_rasters_clipped))
    year_vect[i] = 1
    i += 1
    
    # 
    gt_year = year_vect.copy()
    
    # stacking up the values to have a (m, 5) dimensions gt
    for ind in range(m-1):
        gt_year = np.row_stack([gt_year, year_vect.copy()])
    
    # adding up to the gt list
    gt.append(gt_year)
    
# stacking up the result and making a list with len=m
gt = np.stack(gt)
gt = gt.reshape(gt.shape[0]*gt.shape[1], gt.shape[2])
gt = list(gt)
    
# visualizing some cases
for i in range(5):
    for year in s_rasters_clipped:
        fun.visualize(s_rasters_clipped[year][i])
     
### building the dataset
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

# adding number of sample
n_train = len(datasets["train"])

## building the NN
"""

we build and train the auto-encoder model
the model is made in the functions file

"""

#stores the parameters
args = mock.Mock() 
args.n_epoch = 10
args.batch_size = 64
args.n_channel = 1
args.conv_width = [16,16,32,32,64,32]
args.dconv_width = [32,16,32,16]
args.cuda = 1
args.lr = 0.005
trained_model = fun.train_full(args, datasets)

## visualizing the result
for i in range(1):
    
    # visualizing training raster
    raster = datasets["train"][i]
    fun.visualize(raster)
    
    # visualizing prediction
    pred = trained_model(raster[None,:,:,:].float().cuda()).cpu()
    fun.visualize(pred.detach().numpy().squeeze())
    
### view the embeddings in the model
'''
Now we are going to visualize various embeddings in the model itself
'''

# visualizing embedding inside the model
fun.view_u(datasets["train"], 8, trained_model)

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
threshold = 1.2

# computing change raster
cd_rast, dccode, code1, code2 = fun.change_detection(rast1, rast2, trained_model, threshold)

# visualizing result
fun.visualize(rast1[:,:,:].squeeze(), third_dim = False)
fun.visualize(rast2[:,:,:].squeeze(), third_dim = False)
show((cd_rast.squeeze().cpu()[1,:,:]> 0).float())
show(cd_rast.squeeze().cpu()[1,:,:])   
fun.view_embeddings(dccode)


