# Project hiatus
# script used to pre-process the data
# 16/11/2020
# Cédric BARON

# loading required packages
import rasterio
import os
from shapely.geometry import box
import geopandas as gpd
from rasterio.mask import mask
import random
from shapely.geometry import Polygon
from numpy import save
from rasterio.plot import show
import numpy as np
from fiona.crs import from_epsg
import matplotlib.pyplot as plt

# setting the work directory
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

# importing our functions
import utils as fun

print(
"""

We load the bounding box

""")


## creating the bounding box for cropping our rasters
dir_shp = "data/bdtopobati_frejus"
BuildingsGDF = gpd.read_file(os.path.join(dir_shp, 'bdtopo_bati_1954.shp'))
bbox = BuildingsGDF.total_bounds
    
# setting a bounding box for cropping
glob_minx, glob_miny = bbox[0], bbox[1]
glob_maxx, glob_maxy = bbox[2], bbox[3]
# twe change the miny to avoid the sea
glob_miny = 6265642

# getting global boundaries in the correct format
bbox = box(glob_minx, glob_miny, glob_maxx, glob_maxy)
geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(2154))
coords = fun.getFeatures(geo)

print(
"""

We load the rasters
Then we sample them into a grid with similar resolution

""")

# list rasters
dir_tifs = "data/tifs"
list_files = fun.get_files(dir_tifs)
list_tifs = [name for name in list_files if name[-3:] == "tif"]
list_tifs.sort(reverse=True)

# storing our rasters per year in a dictionary
dict_rasters = {"1954":[], "1966":[], "1970":[], "1978":[], "1989":[]}

# loading the rasters
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
rasters_clipped = fun.clipping_rasters(dict_rasters, boxes)

print(
"""

We now stack up the rasters into 2*128*128 dimension rasters and normalize them
using z-scores and removing minimum altitude for dem

""")

## stacking the result (every raster has two dimensions, color and altitude)
# normalizing
s_rasters_clipped = {}

# list to extract all the dem rasters
rasters_alt = []

for year in rasters_clipped:
    
    # extracting alt and rad
    alt_rasts = [rast[0] for rast in rasters_clipped[year]]
    rad_rasts = [rast[1] for rast in rasters_clipped[year]]
    
    # subtracting min from alt rasters
    alt_rasts_cl = [rast - np.min(fun.get_min(rast[np.nonzero(rast)])) for rast in alt_rasts]
    
    # getting all the altitude rasters for the mean and std
    rasters_alt += alt_rasts_cl.copy()
    
    # replacing data in the dictionary
    rasters_clipped[year] = [np.stack((alt, rad), axis=0) for alt, rad in zip(alt_rasts_cl, rad_rasts)]

# getting the total of rasters
total_rasters_alt = np.stack(rasters_alt, axis=0)

mu_alt = np.mean(total_rasters_alt)
std_alt = np.std(total_rasters_alt)

for year in rasters_clipped:
    
    # extracting alt and rad
    rad_rasts = [rast[1] for rast in rasters_clipped[year]]
    alt_rasts = [rast[0] for rast in rasters_clipped[year]]
    
    # normalizing
    for i in range(len(alt_rasts)):
        
        # boolean of non zeros
        non_zero_alt = np.nonzero(alt_rasts[i])
        non_zero_rad = np.nonzero(rad_rasts[i])
        
        # normalizing
        rad_rasts[i][non_zero_rad] = (rad_rasts[i][non_zero_rad] - np.mean(rad_rasts[i][non_zero_rad])) / np.std(rad_rasts[i][non_zero_rad])
        alt_rasts[i][non_zero_alt] = (alt_rasts[i][non_zero_alt] - mu_alt) / std_alt
                    
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
    
    

print(
"""

Saving the rasters as numpy files

""")

# index for the files name
ind_gt = 0


## saving the rasters
for year in s_rasters_clipped:
    
    for i, np_mat in enumerate(s_rasters_clipped[year]):
        
        # name of the general file
        file = "data/np_data/"+year+"_"
        
        # saving the raster
        save(file+str(i)+'.npy', np_mat)
        
        # saving the gt
        save(file+'gt_'+str(i)+'.npy', gt[ind_gt])
        
        ind_gt += 1

