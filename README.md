# HDAT9900: Dissertation
A repository for my dissertation code, that examines:
1. The Prediction of radiotherapy plan violation from spatial arrangement of target and organ at risk structures using deep learning
2. Standardising 3-dimensional radiotherapy structure data using deep learning.

Both projects have the following layout:

- project
  - data: 
    - raw: raw 3D data files
    - interim: intermediate data such as point clouds
    - processed: features and labels in '.npy' format for the models.
  - notebooks: notebooks explaining the process of the project
    - project notebook: Contains exploration, preprocessing and model deployment
  - output: output for scripts to save to includes model outcomes
  - src: location of python scripts
    - prepare_data: prepares the data for model deployment
    - models: contains machine learning models
 
Both projects use non-Euclidean (PointNet) and Euclidean (3D CNN) based neural networks to learn spatial information from 3D ploygon files (.ply) of radiation therapy plan structures. 

The projects were built using Python 3.7, when using a virtual environment use: ' pip3 install -r requirements.txt ' to download the appropriate python packages for model deployment.

Open3d may also be downloaded for preprocessing the data using: 'pip3 install open3d-python' 

*Data is not included due to ethics and was included in the gitignore file.*
