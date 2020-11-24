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

os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

# importing our functions
import utils as fun

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

# to avoid the sea
glob_miny = 6265642

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

"""

Making an histogram of the data, ignoring zero values

"""

for year in rasters_clipped:
    
    # extracting alt and rad
    alt_rasts = [rast[0] for rast in rasters_clipped[year]]
    rad_rasts = [rast[1] for rast in rasters_clipped[year]]
    
    # getting the total of rasters
    total_rasters_alt = np.stack(alt_rasts, axis=0)
    total_rasters_rad = np.stack(rad_rasts, axis=0)
    
    fig, data = plt.subplots()
    data = plt.hist(total_rasters_rad.flatten(), bins='auto', label='radiometry')
    data = plt.hist(total_rasters_alt.flatten(), bins='auto', label='altitude')
    plt.legend(loc='upper right')
    plt.title("Histogram for year " + year)
    plt.show()

"""

We now stack up the rasters into 2*128*128 dimension rasters and normalize them
using z-scores and removing minimum altitude for mns

"""

## stacking the result (every raster has two dimensions, color and altitude)
# normalizing
s_rasters_clipped = {}

for year in rasters_clipped:
    
    # extracting alt and rad
    alt_rasts = [rast[0] for rast in rasters_clipped[year]]
    rad_rasts = [rast[1] for rast in rasters_clipped[year]]
    
    # subtracting min from alt rasters
    alt_rasts = [rast - np.min(rast) for rast in alt_rasts]
    
    # getting the total of rasters
    total_rasters_alt = np.stack(alt_rasts, axis=0)
    total_rasters_rad = np.stack(rad_rasts, axis=0)
    
    # getting stat values
    mu_alt = np.mean(total_rasters_alt)
    std_alt = np.std(total_rasters_alt)
    mu_rad = np.mean(total_rasters_rad)
    std_rad = np.std(total_rasters_alt)
    
    # normalizing
    for i in range(len(alt_rasts)):
        
        # boolean of non zeros
        non_zero_alt = np.nonzero(alt_rasts[i])
        non_zero_rad = np.nonzero(rad_rasts[i])
        
        # normalizing
        alt_rasts[i][non_zero_alt] = (alt_rasts[i][non_zero_alt] - np.mean(alt_rasts[i][non_zero_alt])) / np.std(alt_rasts[i][non_zero_alt])
        rad_rasts[i][non_zero_rad] = (rad_rasts[i][non_zero_rad] - np.mean(rad_rasts[i][non_zero_rad])) / np.std(rad_rasts[i][non_zero_rad])
                    
                    
    #alt_rasts = [(rast-np.min(rast[np.nonzero(rast)])) / (np.max(rast[np.nonzero(rast)])-np.min(rast[np.nonzero(rast)])) for rast in alt_rasts]
    #rad_rasts = [(rast-np.min(rast[np.nonzero(rast)])) / (np.max(rast[np.nonzero(rast)])-np.min(rast[np.nonzero(rast)])) for rast in rad_rasts]
    
    
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
for i in range(5, 6):
    print(i)
    for year in s_rasters_clipped:
        fun.visualize(s_rasters_clipped[year][i].squeeze(), third_dim=False)
        
"""

Saving the rasters as numpy files

"""

os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

ind_gt = 0

save_rasts = False

if save_rasts:

    ## saving the rasters
    for year in s_rasters_clipped:
        
        i = 1
        
        
        for np_mat in s_rasters_clipped[year]:
            
            file = "data/np_data/"+year+"_"
            save(file+str(i)+'.npy', np_mat)
            
            save(file+'gt_'+str(ind_gt)+'.npy', gt[ind_gt])
            
            ind_gt += 1
            
            i += 1

"""

We check the rasters to get interesting samples for ground truth

"""            

## checking interesting samples for gt
ind = random.randint(0, 900)

print(ind)
    
fun.visualize(s_rasters_clipped["1954"][ind][:,:,:], third_dim=False)
fun.visualize(s_rasters_clipped["1966"][ind][:,:,:], third_dim=False)
fun.visualize(s_rasters_clipped["1970"][ind][:,:,:], third_dim=False)

# interesting sample
sample_id = [121, 833, 127, 592, 851, 107, 480, 700, 45, 465,
             230, 416, 844, 237, 636, 13, 518, 298, 707, 576]

# loading gt boxes
sample_box = [boxes[i] for i in sample_id]
sample_box_c = [sample_box[i][0] for i in range(len(sample_box))]

# list to store pnts in tuples
list_pnt_tuple = []

# converting into list of tuples
for i in range(len(sample_box_c)):
    list_pnt_tuple.append([tuple(pnt) for pnt in sample_box_c[i]["coordinates"][0]])

# loading list of polygons
poly_box = [Polygon(list_pnt_tuple[i]) for i in range(len(list_pnt_tuple))]

# loading as a geo df
gs = gpd.GeoSeries(poly_box)

save_poly = False

if save_poly:
    ## we save the polygons as .shp
    os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")
    
    gs.to_file(filename='./data/GT/GT_poly_2.shp', driver='ESRI Shapefile')



"""

Loading the ground truth (binary values)

"""

os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection/data/GT")

# index of the samples we are interested in
sample_shapes = [121, 833, 127, 592, 851, 107, 480, 700, 45, 465,
                 230, 416, 844, 237, 636, 13, 518, 298, 707, 576]

# loading the GT tifs
# list rasters
list_files = os.listdir()

# sorting the names to have similar order
list_tifs = [name for name in list_files if name[-3:] == "tif"]
list_tifs.sort()

# storing our rasters per year in a dictionary
gt_rasters = {"1966":[], "1970":[], "1978":[], "1989":[]}

# loading the dict for the ground truth
gt_clipped = {"1966":[], "1970":[], "1978":[], "1989":[]}

# loading gt boxes
sample_box = [boxes[i] for i in sample_id]
sample_box_c = [sample_box[i][0] for i in range(len(sample_box))]


# loading the rasters
for file, year in zip(list_tifs, gt_rasters):
    
    ortho = rasterio.open(file)
            
    gt_rasters[year].append(ortho)

for ind in sample_id:
    
    # loading the box
    box = boxes[ind]
    
    for year, prev_year in zip(gt_rasters, list(s_rasters_clipped)[:-1]):
                    
        # cropping the rasters
        out_img, out_transform = mask(dataset=gt_rasters[year][0], all_touched=True,
                                                  shapes=box, crop=True)
        
        
        
        # changing resolution to 128*128
        out_img_resh = fun.regrid(out_img.reshape(out_img.shape[1:]), 128, 128)
        
        # changing values to zero / one
        no_data = out_img_resh < 0
        data = out_img_resh > 0
        out_img_resh[no_data] = 0
        out_img_resh[data] = 1
        
        # loading the corresponding old and new rasters
        alt_old = s_rasters_clipped[prev_year][ind][0,:,:]
        rad_old = s_rasters_clipped[prev_year][ind][1,:,:]
        alt_new = s_rasters_clipped[year][ind][0,:,:]
        rad_new = s_rasters_clipped[year][ind][1,:,:]
        
        # list rasters to stack up
        list_rast = [out_img_resh, alt_old, rad_old, alt_new, rad_new]
        
        stack_rast = np.stack(list_rast, axis=0)
        
        # saving the ground truth and the corresponding raster values
        gt_clipped[year].append(stack_rast)
    

# saving ground truth
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")
    

## saving the rasters
for year in gt_clipped:
    i = 1
    for sample in gt_clipped[year]:
        
        # general name of the file
        file = "data/GT_np/"+year+"_"+str(i)+"_"
        
        # saving the change map
        save(file+"cmap"+'.npy', sample)
        
        i +=1
        
# checking for a given yeqr
year = "1978"

for sample in gt_clipped[year]:
    show(sample[0,:,:])
    
    fun.visualize(sample[1:3,:,:], third_dim=False)
    
    fun.visualize(sample[-2:,:,:], third_dim=False)
    
    break



"""

Loading the ground truth (classes)

"""


os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection/data/GT_class")

# index of the samples we are interested in
sample_shapes = [121, 833, 127, 592, 851, 107, 480, 700, 45, 465,
                 230, 416, 844, 237, 636, 13, 518, 298, 707, 576]


# loading the GT tifs
# list rasters
list_files = os.listdir()

# sorting the names to have similar order
list_tifs = [name for name in list_files if name[-3:] == "tif"]
list_tifs.sort()

# storing our rasters per year in a dictionary
gt_rasters = {"1954":[],"1966":[], "1970":[], "1978":[], "1989":[]}

# loading the dict for the ground truth
gt_clipped = {"1954":[],"1966":[], "1970":[], "1978":[], "1989":[]}

# loading gt boxes
sample_box = [boxes[i] for i in sample_shapes]
sample_box_c = [sample_box[i][0] for i in range(len(sample_box))]

# loading the rasters
for file, year in zip(list_tifs, gt_rasters):
    
    ortho = rasterio.open(file)
            
    gt_rasters[year].append(ortho)

for ind in sample_id:
    
    # loading the box
    box = boxes[ind]
    
    for year in gt_rasters:
                    
        # cropping the rasters
        out_img, out_transform = mask(dataset=gt_rasters[year][0], all_touched=True,
                                                  shapes=box, crop=True)
        
        # changing resolution to 128*128
        out_img_resh = fun.regrid(out_img.reshape(out_img.shape[1:]), 128, 128, "nearest")
        out_img_resh = np.rint(out_img_resh)
        
        # loading corresponding rasters
        alt_new = s_rasters_clipped[year][ind][0,:,:]
        rad_new = s_rasters_clipped[year][ind][1,:,:]
        
        # list rasters to stack up
        list_rast = [out_img_resh, alt_new, rad_new]
        
        stack_rast = np.stack(list_rast, axis=0)
        
        # saving the ground truth 
        gt_clipped[year].append(stack_rast)
    

# saving ground truth
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection/data/GT_np/")
    
save_tifs = False

if save_tifs:
    ## saving the rasters
    for year in gt_clipped:
        i = 1
        for sample in gt_clipped[year]:
            
            # general name of the file
            file = year+"_"+str(i)+"_class"
            
            # saving the change map
            save(file+"cmap"+'.npy', sample)
            
            i +=1
        