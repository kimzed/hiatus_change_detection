# Project hiatus
# main script
# 12/10/2020
# Cédric BARON

# loading required packages
import numpy as np
import rasterio
import os
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
from rasterio.mask import mask
import torch
import mock
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torchvision import transforms
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from osgeo import gdal, ogr
from shapely.geometry import Polygon
from numpy import save
from numpy import load

import imageio

os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")

# importing our functions
import functions as fun
import model as mod

# loading the rasters
load_rasters = True

if load_rasters:
    
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
    
    # loading the files
    for year in s_rasters_clipped:
        for file in list_files:
            if year in file:
                s_rasters_clipped[year].append(load(file))
                
else:
    
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
    
    Converting the building vector data into a raster (groundtruth)
    
    """
    
    # Define pixel_size and NoData value of new raster
    pixel_size = 0.5
    NoData_value = 0
    
    # Filename of input OGR file
    vector_fn = 'bdtopo_bati_1954.shp'
    
    # Filename of the raster Tiff that will be created
    raster_fn = 'ground_truth.tif'
    
    # Open the data source and read in the extent
    source_ds = ogr.Open(vector_fn)
    source_layer = source_ds.GetLayer()
    xmin, xmax, ymin, ymax = source_layer.GetExtent()
    
    # Create the destination data source
    x_res = int((xmax - xmin) / pixel_size)
    y_res = int((ymax - ymin) / pixel_size)
    target_ds = gdal.GetDriverByName('GTiff').Create(raster_fn, x_res, y_res, 1, gdal.GDT_Byte)
    
    # adds coordinates to the point and does a transformation
    target_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))
    
    # get the band and define nodata values
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NoData_value)
    
    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, burn_values=[1])
    
    # get the raster as a numpy array
    mask_arr=target_ds.GetRasterBand(1).ReadAsArray()
    
    # updating the tif
    target_ds.FlushCache()
    
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
    rasters_clipped = {}
    
    # storing some boxes in a list to get interesting gt later
    
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
    
    clipping the vector to get ground truth
    
    """
    
    
    ## clipping the ground truth and putting it into our dictionary
    # creating a new key in the dictionary
    gt_rasters = []
    
    # loading the gt raster
    rast_gt = rasterio.open(raster_fn)
    
    # looping through the patches
    for our_box in boxes:    
        
        # cropping
        out_img, out_transform = mask(dataset=rast_gt, all_touched=True,
                                                  shapes=our_box, crop=True)
        
        # storing in the list
        gt_rasters.append(out_img)
    
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
            alt_rasts[i][non_zero_alt] = (alt_rasts[i][non_zero_alt] - alt_rasts[i][non_zero_alt].mean()) / alt_rasts[i][non_zero_alt].std() + 10
            rad_rasts[i][non_zero_rad] = (rad_rasts[i][non_zero_rad] - rad_rasts[i][non_zero_rad].mean()) / rad_rasts[i][non_zero_rad].std() + 10
                        
                        
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
    
    ## saving the rasters
    for year in s_rasters_clipped:
        
        i = 1
        
        for np_mat in s_rasters_clipped[year]:
            
            file = "data/np_data/"+year+"_"
            save(file+str(i)+'.npy', np_mat)
            
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
    fun.visualize(s_rasters_clipped["1989"][ind][:,:,:], third_dim=False)
    
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
    
    ## we save the polygons as .shp
    os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")
    
    gs.to_file(filename='./data/GT/GT_poly_2.shp', driver='ESRI Shapefile')


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

## we need to combine images and labels for the discriminator
train_data = []

for i in range(len(datasets["train"])):
   train_data.append([datasets["train"][i], datasets["gt_train"][i]])

# adding number of sample
n_train = len(datasets["train"])

## building the NN
"""

we build and train the auto-encoder model
the model is made in the functions file

"""

## loading the model in case we have it already
load_model = True

## working with tensorboard
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection")
writer = SummaryWriter('runs/0911_7_deftest')

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
    args.batch_size = 32
    args.n_channel = 1
    args.conv_width = [8,4,8,8,16,8]
    args.dconv_width = [8,4,8,4]
    args.cuda = 1
    args.lr = 0.00002
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
    pred = trained_model(raster[None,:,:,:].float().cuda())[0].cpu()
    fun.visualize(pred.detach().numpy().squeeze(), third_dim=False)
    
### view the embeddings in the model
'''

Now we are going to visualize various embeddings in the model itself

'''


fun.view_u(datasets["train"], trained_model, random.randint(0, 900))

# visualizing embedding inside the model
fun.view_u(s_rasters_clipped["1966"], trained_model, 531)
fun.view_u(s_rasters_clipped["1970"], trained_model, 531)

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
dccode, code1, code2 = fun.change_detection(rast1, rast2, trained_model, threshold)

# visualizing result
fun.visualize(rast1[:,:,:].squeeze(), third_dim = False)
fun.visualize(rast2[:,:,:].squeeze(), third_dim = False)
fun.view_embeddings(dccode)

"""

Performing change detection analysis on actual data

"""
ind = random.randint(0, 1500)

## running cd model
rast1 = s_rasters_clipped["1954"][482][None,:,:,:]
rast2 = s_rasters_clipped["1966"][482][None,:,:,:]

ind = random.randint(0, 900)

fun.visualize(s_rasters_clipped["1954"][482][:,:,:], third_dim=False)
fun.visualize(s_rasters_clipped["1966"][482][:,:,:], third_dim=False)

    
threshold = 1.5

# computing change raster
dccode, code1, code2 = fun.change_detection(rast1, rast2, trained_model, threshold)

# visualizing result
fun.visualize(rast1[:,:,:].squeeze(), third_dim = False)
fun.visualize(rast2[:,:,:].squeeze(), third_dim = False)
fun.view_embeddings(dccode)

fun.view_embeddings(dccode)

rast_fin = dccode[0,0,:,:].detach().squeeze().cpu()

for i in range(1, 8):
    
    rast = dccode[0,i,:,:].detach().squeeze().cpu()
    
    rast
    rast_fin += rast
    
bin_rast = (rast_fin.abs() > 0).float()
    

for i in range(32):
    
    rast = code1[0,i,:,:].detach().squeeze().cpu()
    
    show(rast)
    
    rast = code2[0,i,:,:].detach().squeeze().cpu()
    
    show(rast)
