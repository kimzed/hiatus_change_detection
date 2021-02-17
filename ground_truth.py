# Project hiatus
# script used for the ground truth
# 17/02/2021
# Cédric BARON

"""

We check the rasters to get interesting samples for ground truth and save the
patches as .shp

"""            

## checking interesting samples for gt
ind = random.randint(0, 900)

print(ind)
    
fun.visualize(s_rasters_clipped["1954"][ind][:,:,:], third_dim=False)
fun.visualize(s_rasters_clipped["1966"][ind][:,:,:], third_dim=False)
fun.visualize(s_rasters_clipped["1970"][ind][:,:,:], third_dim=False)

# interesting sample
sample_id = [121, 833, 127, 592, 851, 107, 480, 700, 45, 465,
             230, 416, 844, 237, 636, 13, 518, 298, 707, 576,
             40, 97, 212, 391, 402, 428, 464, 515, 565, 581, 689,
             302, 153, 466, 482, 105, 341, 337, 782, 398, 153,
             88, 342, 318, 126, 481, 554, 138, 447, 189, 224]

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
    
    gs.to_file(filename='./data/GT/GT_poly_3.shp', driver='ESRI Shapefile')

"""

Loading the ground truth (classes)

"""

# change working directory
os.chdir("/home/adminlocal/Bureau/GIT/hiatus_change_detection/data/GT_class")

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
sample_box = [boxes[i] for i in sample_id]
sample_box_c = [sample_box[i][0] for i in range(len(sample_id))]


# loading the rasters
for file, year in zip(list_tifs, gt_rasters):
    
    ortho = rasterio.open(file)
            
    gt_rasters[year].append(ortho)

# loading the rasters
for ind in sample_id:
    
    # loading the box
    box = boxes[ind]
    
    for year in gt_rasters:
                    
        # cropping the rasters
        out_img, out_transform = rasterio.mask.mask(dataset=gt_rasters[year][0], all_touched=True,
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
        