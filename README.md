# Unsupervized Temporal Domain Adaptation for Change Detection in Historic 3D Data Time-Series

Welcome to the PyTorch implementation of the change detection project conducted at the Lastig lab, IGN. This project uses advanced machine learning techniques to detect temporal changes in the landscape of Fréjus, France, from aerial photos throughout the 20th century.

Author: Cédric BARON

# Code structure

```./evaluation_models/``` stores various models that can be loaded in the script.

```frejus_dataset.py``` is a script used to load the dataset.

```ground_truth.py``` is an archive of the code used to produce the ground truth.

```model_evaluation.py``` loads a model, runs various visualization on it and assesses its performances.

```pre_processing.py``` is the script used to pre_process the raw rasters.

To compute a model and perform a general assesment, the ```main.py``` and the other scripts are used.

# Requirements

For the coding environment I used an environment made by my university, which entails various geo-packages. You can run this line of code in your console:

```
wget https://raw.githubusercontent.com/GeoScripting-WUR/InstallLinuxScript/master/user/environment.yml

conda env create -f environment.yml
```

# Running the code

To train a model with specific settings use:

```python main.py --epochs 25 --lr 0.025```

Note: Check main.py for a full list of command-line arguments and their descriptions.

# Evaluation

Models are stored in ./evaluation_models/. To evaluate a specific model, enable the evaluation flag in main.py and specify the model's name.


