# Hiatus Change Detection

Unsupervised temporal domain adaptation for change detection in historic 3D aerial imagery time-series.

## Overview

A PyTorch implementation of change detection on 3D historic aerial imagery of Frejus, France, developed at the LASTIG lab (IGN). The project applies unsupervised domain adaptation techniques to detect urban and landscape changes across temporal sequences of aerial reconstructions, without requiring labeled data for each time period.

## Tech Stack

- **Deep Learning:** PyTorch
- **Geospatial:** GDAL, Rasterio, GeoPandas
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib
- **Environment:** Conda

## Project Structure

```
hiatus_change_detection/
├── main.py                  # Training entry point
├── frejus_dataset.py        # Dataset loader for aerial imagery
├── pre_processing.py        # Raw raster preprocessing
├── model_evaluation.py      # Model loading, visualization, performance assessment
├── ground_truth.py          # Ground truth generation (archive)
└── evaluation_models/       # Saved model checkpoints
```

## Usage

```bash
wget https://raw.githubusercontent.com/GeoScripting-WUR/InstallLinuxScript/master/user/environment.yml
conda env create -f environment.yml

python main.py --epochs 25 --lr 0.025
```

## Context

MSc thesis project at IGN (Institut national de l'information geographique et forestiere), LASTIG lab. Investigates unsupervised change detection methods on 3D photogrammetric reconstructions from historical aerial photography campaigns over Frejus, France.

By Cedric Baron.
