#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:27:02 2019

@author: philliphungerford

Purpose: Data preparation file for dissertation standardisation (aim 2)
"""

# =============================================================================
# 0. Dependencies
# =============================================================================

# Install dependencies
import os
import numpy as np
import pandas as pd
import open3d
from matplotlib import pyplot as plt


# =============================================================================
# 1. Functions
# =============================================================================

def max_point_calculator(max_num_only="y", ignore_body="y"):
    """
    This will read the file directory, list all files, find the patient numbers
    Then for a given patient, load all of their files and save the number of points
    into an array of shape (ID, number of points) with a length the number of patients
    """
    # dependencies
    import glob
    import os
    from open3d import read_point_cloud
    import numpy as np
    import re
    os.chdir('../0_data/prostate-no-nodes')
    # This will read the file directory, list all files, find the patient numbers
    # Then for a given patient, load all of their files and merge them into an array
    # File directory is current
    path = '.'
    files = os.listdir(path)  # lists all files in the current directory
    pt_num = []
    for file in files:
        regex = re.compile(r'\d+')
        # for each file extract the number which corresponds to a patient
        for x in regex.findall(file):
            number = int(x)
            pt_num.append(number)
    # turn that list into a set such that each item in the list is now a patient
    pt_nums = set(pt_num)
    pt_nums = [i for i in pt_nums]
    
    ############################################################################
    # Phase 2
    
    # Create place holder for patient name and number of points for that patient
    points_per_patient = np.ndarray(shape=(len(pt_nums),2))
    
    # Create loop 
    for count, patient in enumerate(pt_nums):
        # Instead of this, lets get a list of the file names and then iterate and scan
        patient_files = glob.glob('Mesh_PtNum-' + str(patient) + '*.ply')  # 5433
        
        # create a list to put our points per file in
        temp = []
        for file in patient_files:
            if ignore_body=="y":
                
                if file != 'Mesh_PtNum-' + str(patient) + '-BODY.ply':
                    tmp = read_point_cloud(file)
                    tmp = np.asarray(tmp.points)
                    temp.append(tmp)
            else:
                tmp = read_point_cloud(file)
                tmp = np.asarray(tmp.points)
                temp.append(tmp)
                    

        all_points = np.vstack(temp)
        points_per_patient[count,0] = patient # column 1 is the current patient
        points_per_patient[count,1] = len(all_points) # column 2 is the number of points
        # print("Patient ", patient, "has ", len(all_points), "points.")
    # Change the directory back to local 
    os.chdir('../../1_code')
    
    max_num_points = int(min(points_per_patient[:,1]))
    print("The lowest number of points you can use is: ",max_num_points )
    
    if max_num_only == "y":
        return max_num_points
    else:
        return max_num_points, points_per_patient

		
def max_point_calculator_organ():
    """
    This will read the file directory, list all files, find the patient numbers
    Then for a given patient, load all of their files and save the number of points
    into an array of shape (ID, number of points) with a length the number of patients
    """
    # dependencies
    import glob
    import os
    from open3d import read_point_cloud
    import numpy as np
    import re
    os.chdir('../0_data/prostate-no-nodes')
    # This will read the file directory, list all files, find the patient numbers
    # Then for a given patient, load all of their files and merge them into an array
    # File directory is current
    path = '.'
    files = os.listdir(path)  # lists all files in the current directory
    pt_num = []
    for file in files:
        regex = re.compile(r'\d+')
        # for each file extract the number which corresponds to a patient
        for x in regex.findall(file):
            number = int(x)
            pt_num.append(number)
    # turn that list into a set such that each item in the list is now a patient
    pt_nums = set(pt_num)
    pt_nums = [i for i in pt_nums]
    
    ############################################################################
    # Phase 2
    
    max_prostate=10000 
    max_bladder=10000
    max_feml=10000
    max_femr=10000
    max_rectum=10000
    max_body=10000
    
    print("The lowest points for:")
    
    # Create loop 
    for patient in pt_nums:
        # Instead of this, lets get a list of the file names and then iterate and scan
        prostate = read_point_cloud('Mesh_PtNum-' + str(patient) + '-PTVHD.ply')
        bladder = read_point_cloud('Mesh_PtNum-' + str(patient) + '-Bladder.ply')
        feml = read_point_cloud('Mesh_PtNum-' + str(patient) + '-FemoralHeadL.ply')
        femr = read_point_cloud('Mesh_PtNum-' + str(patient) + '-FemoralHeadR.ply')
        rectum = read_point_cloud('Mesh_PtNum-' + str(patient) + '-Rectum.ply')
        body = read_point_cloud('Mesh_PtNum-' + str(patient) + '-BODY.ply')

        # 2.Convert point cloud format to cartesian coordinates
        prostate = np.asarray(prostate.points)
        if len(prostate) < max_prostate:
            max_prostate = len(prostate)
            print("Prostate is now: ", max_prostate)
        
        bladder = np.asarray(bladder.points)
        if len(bladder) < max_bladder:
            max_bladder = len(bladder)
            print("Bladder is now: ", max_bladder)
        
        feml = np.asarray(feml.points)
        if len(feml) < max_feml:
            max_feml = len(feml)
            print("Left Fem is now: ", max_feml)
            
        femr = np.asarray(femr.points)
        if len(femr) < max_femr:
            max_femr = len(femr)
            print("Right Fem is now: ", max_femr)
            
        rectum = np.asarray(rectum.points)
        if len(rectum) < max_rectum:
            max_rectum = len(rectum)
            print("Rectum is now: ", max_rectum)
            
        body = np.asarray(body.points)
        if len(body) < max_body:
            max_body = len(body)
            print("Body is now: ", max_body)
    
    print("Calculating the lowest number of points for all organs...")              
    lowest = [max_prostate, max_bladder, max_feml, max_femr, max_rectum, max_body]
    print("Done! It is ", min(lowest))
    # Change the directory back to local 
    os.chdir('../../1_code')
    return lowest
	
	
def import_data(folder, name, points_per_sample):
    """
    This will read the file directory, list all files, find the patient numbers
    Then for a given patient, load all of their files and merge them into an array
    """
    # dependencies
    import glob
    import os
    import data_prep
    from open3d import read_point_cloud
    import numpy as np
    import re
    os.chdir('../0_data/' + folder)
    # This will read the file directory, list all files, find the patient numbers
    # Then for a given patient, load all of their files and merge them into an array
    # File directory is current
    path = '.'
    files = os.listdir(path)  # lists all files in the current directory
    pt_num = []
    for file in files:
        regex = re.compile(r'\d+')
        # for each file extract the number which corresponds to a patient
        for x in regex.findall(file):
            number = int(x)
            pt_num.append(number)
    # turn that list into a set such that each item in the list is now a patient
    pt_nums = set(pt_num)
    pt_nums = [i for i in pt_nums]
    all_patients = np.ndarray(shape=(len(pt_nums), points_per_sample, 3))
    
    print("Importing...")
    for count, patient in enumerate(pt_nums):
        # Instead of this, lets get a list of the file names and then iterate and scan
        patient_files = glob.glob('Mesh_PtNum-' + str(patient) + '*.ply')  # 5433
        
        # create a list to put our points per file in
        temp = []
        for file in patient_files:
            
            if file != 'Mesh_PtNum-' + str(patient) + '-BODY.ply':
                tmp = read_point_cloud(file)
                tmp = np.asarray(tmp.points)
                temp.append(tmp)

        all_points = np.vstack(temp)
        downsampled = data_prep.downsample(all_points, points_per_sample)
        all_patients[count] = downsampled
    
    print("Saving...")
    np.save('../../2_pipeline/' + name + '.npy', all_patients)
    np.save('../../2_pipeline/' + name + '-order.npy', pt_nums)
    
    os.chdir('../../1_code')
    print("Done!")
    
    return all_patients, pt_nums

	
# Model 1 ---------------------------------------------------------------------
# Model 1 & 2 - 3140 data with each organ downsampled by points (628 points per organ)

import numpy as np

# farthest point calculation
def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)

def downsample(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts

# Re-download model that enters another column that specifies if point is PTVHD
def import_data_xyz(folder, name, points_per_sample, number_of_organs):
    """
    This will read the file directory, list all files, find the patient numbers
    Then for a given patient, load all of their files and merge them into an array
    """
    # dependencies
    import glob
    import os
    import data_prep
    from open3d import read_point_cloud
    import numpy as np
    import re
    os.chdir('../1_code')
    os.chdir('../0_data/' + folder)
    # This will read the file directory, list all files, find the patient numbers
    # Then for a given patient, load all of their files and merge them into an array
    # File directory is current
    path = '.'
    files = os.listdir(path)  # lists all files in the current directory
    pt_num = []
    for file in files:
        regex = re.compile(r'\d+')
        # for each file extract the number which corresponds to a patient
        for x in regex.findall(file):
            number = int(x)
            pt_num.append(number)
    # turn that list into a set such that each item in the list is now a patient
    pt_nums = set(pt_num)
    pt_nums = [i for i in pt_nums]
    all_patients = np.ndarray(shape=(len(pt_nums), int(points_per_sample/number_of_organs)*number_of_organs, 3))
    ############################################################################
    
    for count, patient in enumerate(pt_nums):
        
        print('Importing patient ', patient)
        # Read ply files to point cloud format
        prostate = read_point_cloud('Mesh_PtNum-' + str(patient) + '-PTVHD.ply')
        bladder = read_point_cloud('Mesh_PtNum-' + str(patient) + '-Bladder.ply')
        feml = read_point_cloud('Mesh_PtNum-' + str(patient) + '-FemoralHeadL.ply')
        femr = read_point_cloud('Mesh_PtNum-' + str(patient) + '-FemoralHeadR.ply')
        rectum = read_point_cloud('Mesh_PtNum-' + str(patient) + '-Rectum.ply')
        # body = read_point_cloud('Mesh_PtNum-' + str(patient) + '-BODY.ply')

        # Convert point cloud format to cartesian coordinates
        prostate = np.asarray(prostate.points)
        bladder = np.asarray(bladder.points)
        feml = np.asarray(feml.points)
        femr = np.asarray(femr.points)
        rectum = np.asarray(rectum.points)
        # body = np.asarray(body.points)
        
        # downsample point cloud
        prostate = downsample(prostate, int(points_per_sample/number_of_organs))
        bladder = downsample(bladder, int(points_per_sample/number_of_organs))
        feml = downsample(feml, int(points_per_sample/number_of_organs))
        femr = downsample(femr, int(points_per_sample/number_of_organs))
        rectum = downsample(rectum, int(points_per_sample/number_of_organs))
        # body = downsample(body, int(points_per_sample/number_of_organs))

        # Combine them 
        combined = np.concatenate((prostate,bladder,feml,femr,rectum)) # body
        # combined will have shape = (p, 3) where 3 is x,y,z

        # add downsampled and labelled patient to all patients array
        all_patients[count] = combined
    
    print("Saving...")
    np.save('../../2_pipeline/' + name + '.npy', all_patients)
    np.save('../../2_pipeline/' + name + '-order.npy', pt_nums)
    
    os.chdir('../../1_code')
    print("Done!")
    
    return all_patients, pt_nums
	
# Model 3: PointNet with added 'l' column --------------------------------------
import numpy as np

# farthest point calculation
def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)

def downsample(pts, K):
    farthest_pts = np.zeros((K, 3))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts

# Re-download model that enters another column that specifies if point is PTVHD
def import_data_xyzl(folder, name, points_per_sample, number_of_organs):
    """
    This will read the file directory, list all files, find the patient numbers
    Then for a given patient, load all of their files and merge them into an array
    """
    # dependencies
    import glob
    import os
    import data_prep
    from open3d import read_point_cloud
    import numpy as np
    import re
    os.chdir('../../1_code')
    os.chdir('../0_data/' + folder)
    # This will read the file directory, list all files, find the patient numbers
    # Then for a given patient, load all of their files and merge them into an array
    # File directory is current
    path = '.'
    files = os.listdir(path)  # lists all files in the current directory
    pt_num = []
    for file in files:
        regex = re.compile(r'\d+')
        # for each file extract the number which corresponds to a patient
        for x in regex.findall(file):
            number = int(x)
            pt_num.append(number)
    # turn that list into a set such that each item in the list is now a patient
    pt_nums = set(pt_num)
    pt_nums = [i for i in pt_nums]
    all_patients = np.ndarray(shape=(len(pt_nums), int(points_per_sample/number_of_organs)*number_of_organs, 4))
    ############################################################################
    
    for count, patient in enumerate(pt_nums):
        
        print('Importing patient ', patient)
        # Read ply files to point cloud format
        prostate = read_point_cloud('Mesh_PtNum-' + str(patient) + '-PTVHD.ply')
        bladder = read_point_cloud('Mesh_PtNum-' + str(patient) + '-Bladder.ply')
        feml = read_point_cloud('Mesh_PtNum-' + str(patient) + '-FemoralHeadL.ply')
        femr = read_point_cloud('Mesh_PtNum-' + str(patient) + '-FemoralHeadR.ply')
        rectum = read_point_cloud('Mesh_PtNum-' + str(patient) + '-Rectum.ply')
        # body = read_point_cloud('Mesh_PtNum-' + str(patient) + '-BODY.ply')

        # Convert point cloud format to cartesian coordinates
        prostate = np.asarray(prostate.points)
        bladder = np.asarray(bladder.points)
        feml = np.asarray(feml.points)
        femr = np.asarray(femr.points)
        rectum = np.asarray(rectum.points)
        # body = np.asarray(body.points)
        
        # downsample point cloud
        prostate = downsample(prostate, int(points_per_sample/number_of_organs))
        bladder = downsample(bladder, int(points_per_sample/number_of_organs))
        feml = downsample(feml, int(points_per_sample/number_of_organs))
        femr = downsample(femr, int(points_per_sample/number_of_organs))
        rectum = downsample(rectum, int(points_per_sample/number_of_organs))
        # body = downsample(body, int(points_per_sample/number_of_organs))
        
        # Add dimension to signify that prostate is target
        prostate=np.insert(prostate, 3, 1, axis=1)
        bladder=np.insert(bladder, 3, 0, axis=1)
        feml=np.insert(feml, 3, 0, axis=1)
        femr=np.insert(femr, 3, 0, axis=1)
        rectum=np.insert(rectum, 3, 0, axis=1)
        # body=np.insert(body, 3, 0, axis=1)
        
        # Combine them 
        combined = np.concatenate((prostate,bladder,feml,femr,rectum)) # body
        # combined will have shape = (p, 4) where 4 is x,y,z,l

        # add downsampled and labelled patient to all patients array
        all_patients[count] = combined
    
    print("Saving...")
    np.save('../../2_pipeline/' + name + '.npy', all_patients)
    np.save('../../2_pipeline/' + name + '-order.npy', pt_nums)
    
    os.chdir('../../1_code')
    print("Done!")
    
    return all_patients, pt_nums
	
# Model 3: PointNet with added 'l' column --------------------------------------
# Re-download model that enters another column that specifies if point is PTVHD
def import_data_4D(folder, name, points_per_sample):
    """
    This will read the file directory, list all files, find the patient numbers
    Then for a given patient, load all of their files and merge them into an array
    """
    # dependencies
    import glob
    import os
    import data_prep
    from open3d import read_point_cloud
    import numpy as np
    import re
    os.chdir('../0_data/' + folder)
    # This will read the file directory, list all files, find the patient numbers
    # Then for a given patient, load all of their files and merge them into an array
    # File directory is current
    path = '.'
    files = os.listdir(path)  # lists all files in the current directory
    pt_num = []
    for file in files:
        regex = re.compile(r'\d+')
        # for each file extract the number which corresponds to a patient
        for x in regex.findall(file):
            number = int(x)
            pt_num.append(number)
    # turn that list into a set such that each item in the list is now a patient
    pt_nums = set(pt_num)
    pt_nums = [i for i in pt_nums]
    all_patients = np.ndarray(shape=(len(pt_nums), int(points_per_sample/5)*5, 3, 3))
    ############################################################################
    
    for count, patient in enumerate(pt_nums):
        
        print('Importing patient ', patient)
        # 1.Read ply files to point cloud format
        prostate = read_point_cloud('Mesh_PtNum-' + str(patient) + '-PTVHD.ply')
        bladder = read_point_cloud('Mesh_PtNum-' + str(patient) + '-Bladder.ply')
        feml = read_point_cloud('Mesh_PtNum-' + str(patient) + '-FemoralHeadL.ply')
        femr = read_point_cloud('Mesh_PtNum-' + str(patient) + '-FemoralHeadR.ply')
        rectum = read_point_cloud('Mesh_PtNum-' + str(patient) + '-Rectum.ply')
        # body = read_point_cloud('Mesh_PtNum-' + str(patient) + '-BODY.ply')

        # 2.Convert point cloud format to cartesian coordinates
        prostate = np.asarray(prostate.points)
        bladder = np.asarray(bladder.points)
        feml = np.asarray(feml.points)
        femr = np.asarray(femr.points)
        rectum = np.asarray(rectum.points)
        # body = np.asarray(body.points)
        
        # 3.Downsample point cloud
        points_per_organ = int(points_per_sample/5)
        prostate = downsample(prostate, points_per_organ)
        bladder = downsample(bladder, points_per_organ)
        feml = downsample(feml, points_per_organ)
        femr = downsample(femr, points_per_organ)
        rectum = downsample(rectum, points_per_organ)
        # body = downsample(body, int(points_per_sample/6))
        
        # 4.Add dimension to signify that prostate is target
        
        # 4.1 Create 4D placeholder
        prostate_coloured = np.ndarray((points_per_organ, 3, 3))
        bladder_coloured = np.ndarray((points_per_organ, 3, 3))
        rectum_coloured = np.ndarray((points_per_organ, 3, 3))
        feml_coloured = np.ndarray((points_per_organ, 3, 3))
        femr_coloured = np.ndarray((points_per_organ, 3, 3))
        # body_coloured = np.ndarray((points_per_organ, 3, 3))
        
        # Create function that adds colour dimension to data
        from matplotlib.pyplot import cm

        # Target volume is red
        def add_rgb_prostate(array):
            scaler_map = cm.ScalarMappable(cmap="Reds")
            array = scaler_map.to_rgba(array)[:, : -1]
            return array
        # Add red to prostate and enter to placeholder
        for i in range(prostate.shape[0]):
            prostate_coloured[i] = add_rgb_prostate(prostate[i])

        # Other organs are blue and entered to placeholder
        def add_rgb_organs(array):
            scaler_map = cm.ScalarMappable(cmap="Blues")
            array = scaler_map.to_rgba(array)[:, : -1]
            return array
        # Add blue to other organs
        for i in range(prostate.shape[0]):
            bladder_coloured[i] = add_rgb_organs(bladder[i])
            rectum_coloured[i] = add_rgb_organs(rectum[i])
            feml_coloured[i] = add_rgb_organs(feml[i])
            femr_coloured[i] = add_rgb_organs(femr[i])
            # body_coloured[i] = add_rgb_prostate(prostate[i])
        
        # 5.Combine them 
        combined = np.concatenate((prostate_coloured,bladder_coloured,feml_coloured,femr_coloured,rectum_coloured))
        
        # 6.Add downsampled and labelled patient to all patients array
        all_patients[count] = combined
    
    print("Saving...")
    np.save('../../2_pipeline/' + name + '-4D.npy', all_patients)
    np.save('../../2_pipeline/' + name + '-4D-order.npy', pt_nums)
    
    os.chdir('../../1_code')
    print("Done!")
    
    return all_patients, pt_nums
# =============================================================================
# 2. Prepare data
# =============================================================================

if __name__ == '__main__':
    dataset12, ptnums12 = import_data_xyz(folder='prostate-no-nodes', name='3140-xyz', points_per_sample=3140, number_of_organs=5)
	dataset3, ptnums3 = import_data_xyzl(folder='prostate-no-nodes', name='3768-body-xyzl', points_per_sample=3768, number_of_organs=6)
	dataset4, ptnums4 = import_data_4D(folder='prostate-no-nodes', name='2048', points_per_sample=2048)