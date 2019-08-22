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

"""
VoxelGrid Class
"""
from matplotlib import pyplot as plt
import numpy as np
#from ..plot import plot_voxelgrid

class VoxelGrid(object):
    def __init__(self, points, x_y_z=[1, 1, 1], bb_cuboid=False, build=True):
        """
        Parameters
        ----------         
        points: (N,3) ndarray
                The point cloud from which we want to construct the VoxelGrid.
                Where N is the number of points in the point cloud and the
                second dimension represents the x, y and z coordinates of each
                point.
        
        x_y_z:  list
                The segments in which each axis will be divided.
                x_y_z[0]: x axis 
                x_y_z[1]: y axis 
                x_y_z[2]: z axis

        bb_cuboid(Optional): bool
                If True(Default):   
                    The bounding box of the point cloud will be adjusted
                    in order to have all the dimensions of equal length.                
                If False:
                    The bounding box is allowed to have dimensions of different
                    sizes.
        """
        self.points = points
        xyzmin = np.min(points, axis=0) - 0.001
        xyzmax = np.max(points, axis=0) + 0.001

        if bb_cuboid:
            #: adjust to obtain a  minimum bounding box with all sides of equal
            # length
            diff = max(xyzmax-xyzmin) - (xyzmax-xyzmin)
            xyzmin = xyzmin - diff / 2
            xyzmax = xyzmax + diff / 2
        self.xyzmin = xyzmin
        self.xyzmax = xyzmax

        segments = []
        shape = []

        for i in range(3):
            # note the +1 in num 
            if type(x_y_z[i]) is not int:
                raise TypeError("x_y_z[{}] must be int".format(i))
            s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1),\
                                  retstep=True)
            segments.append(s)
            shape.append(step)
        
        self.segments = segments
        self.shape = shape
        self.n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]
        self.n_x = x_y_z[0]
        self.n_y = x_y_z[1]
        self.n_z = x_y_z[2]
        self.id = "{},{},{}-{}".format(x_y_z[0], x_y_z[1], x_y_z[2], bb_cuboid)
        if build:
            self.build()

    def build(self):
        structure = np.zeros((len(self.points), 4), dtype=int)
        structure[:,0] = np.searchsorted(self.segments[0], self.points[:,0])- 1
        structure[:,1] = np.searchsorted(self.segments[1], self.points[:,1])- 1
        structure[:,2] = np.searchsorted(self.segments[2], self.points[:,2])- 1
        # i = ((y * n_x) + x) + (z * (n_x * n_y))
        structure[:,3] = ((structure[:,1] * self.n_x) + structure[:,0])\
        + (structure[:,2] * (self.n_x * self.n_y)) 
        
        self.structure = structure
        vector = np.zeros(self.n_voxels)
        count = np.bincount(self.structure[:,3])
        vector[:len(count)] = count
        self.vector = vector.reshape(self.n_z, self.n_y, self.n_x)

 
    def plot(self, d=2, cmap="Oranges", axis=False):

        if d == 2:
            fig, axes= plt.subplots(int(np.ceil(self.n_z / 4)),4,figsize=(8,8))
            plt.tight_layout()
            for i,ax in enumerate(axes.flat):
                if i >= len(self.vector):
                    break
                im = ax.imshow(self.vector[i], cmap=cmap, interpolation="none")
                ax.set_title("Level " + str(i))
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label('NUMBER OF POINTS IN VOXEL')
        elif d == 3:
            return plot_voxelgrid(self, cmap=cmap, axis=axis)
			
def import_data(folder, name):
    """
    This will read the file directory, list all files, find the patient numbers
    Then for a given patient, load all of their files and merge them into an array
    """
    # dependencies
    import glob
    import os
    from open3d import read_point_cloud
    import numpy as np
    import re
    os.chdir('../data/' + folder)
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
    all_patients = []
    
    print("Importing...")
    for count, patient in enumerate(pt_nums):
        # Instead of this, lets get a list of the file names and then iterate and scan
        patient_files = glob.glob('Mesh_PtNum-' + str(patient) + '*.ply')
        
        # create a list to put our points per file in
        temp = []
        for file in patient_files:
            tmp = read_point_cloud(file)
            tmp = np.asarray(tmp.points)
            temp.append(tmp)

        all_points = np.vstack(temp)
        all_patients.append(all_points)
    
    print("Saving...")
    os.chdir('../processed/')
    np.save(name + '.npy', all_patients)
    np.save(name + '-order.npy', pt_nums)
    os.chdir('../../notebooks')
    print("Done!")
    
    return all_patients, pt_nums
	
def prepare_labels():
	'''
	This function reads the csv data, extracts the patients who have a plan violation saves the labels returning the clean numpy array for labels
	'''
	# Labels 
	labels = pd.read_csv("../data/labels_n.csv")
	selection = labels.loc[labels['Group'] == '1. pro def no nodes']

	# Subset data by ID and any violations
	violations = selection[['ID', 'Bla/Rec ANY']].copy()
	violations = violations.fillna(0)
	violations = violations.rename(columns={'Bla/Rec ANY':'violation'})
	violations.astype('int64')
	
	# Turn into a dataframe to merge with label data and give it the same column name
	patient_order = pd.DataFrame(patient_order)
	patient_order.columns=["ID"]

	# Merge 2 lists based on column values (matching violation to patient based on order)
	labelzz = pd.merge(patient_order, violations, on='ID')
	# just get the violations in the same order
	y = labelzz[['violation']].values
	
	#turn the list of lists into just one list of 0/1 values
	y = [item for sublist in y for item in sublist]
	y = np.array(y)
	
	return(y)

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


if __name__ == "__main__":
	# Gather point clouds
	pointclouds, patient_order = import_data(name='patientpointclouds', folder='raw/')
	voxel_data = np.ndarray(shape = (len(pointclouds), 16,16,16))
	
	# Voxelise the data
	for i in range(len(pointclouds)):
		plan = pointclouds[i]
		tmp = VoxelGrid(plan, x_y_z=[h,w,d], bb_cuboid=False, build=True)
		tmp = tmp.vector
		voxel_data[i] = tmp
	
	# Downsample point clouds for PointNet model
	pointnet_data = np.ndarray(shape = (len(pointclouds), 1024, 3))
	for i in range(len(pointclouds)):
		plan = pointclouds[i]
		tmp = downsample(plan, 1024)
		pointnet_data[i] = tmp
		
	# save labels
	# save
	np.save('data/processed/X_pointnet.npy', pointnet_data)
	np.save('../data/processed/X_cnn.npy', voxel_data)
	np.save('../data/processed/y.npy', y)