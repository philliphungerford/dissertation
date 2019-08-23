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
  - output: output for scripts to save to includes model outcomes
  - src: location of python scripts
    - prepare_data: prepares the data for model deployment
    - models: contains machine learning models
 
 The 'data' folder should be placed in the project/data/raw directory.
