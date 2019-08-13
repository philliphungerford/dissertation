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
# Voxel Grid Class ------------------------------------------------------------

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
            diff = max(xyzmax - xyzmin) - (xyzmax - xyzmin)
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
            s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
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
		structure[:, 0] = np.searchsorted(self.segments[0], self.points[:, 0]) - 1
		structure[:, 1] = np.searchsorted(self.segments[1], self.points[:, 1]) - 1
		structure[:, 2] = np.searchsorted(self.segments[2], self.points[:, 2]) - 1
		# i = ((y * n_x) + x) + (z * (n_x * n_y))
		structure[:, 3] = ((structure[:, 1] * self.n_x) + structure[:, 0]) \
						  + (structure[:, 2] * (self.n_x * self.n_y))

		self.structure = structure
		vector = np.zeros(self.n_voxels)
		count = np.bincount(self.structure[:, 3])
		vector[:len(count)] = count
		self.vector = vector.reshape(self.n_z, self.n_y, self.n_x)


	def plot(self, d=2, cmap="Oranges", axis=False):
		if d == 2:
			fig, axes = plt.subplots(int(np.ceil(self.n_z / 4)), 4, figsize=(8, 8))
			plt.tight_layout()
			for i, ax in enumerate(axes.flat):
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
	

# Create function for CNN features --------------------------------------------

def get_features(voxel_size, save, binary=False):

    # 1. Get point data from structures ---------------------------------------
    # 1.1. Get a list of the structure files in raw directory
    files = os.listdir('../data/raw/')
    # 1.2. Create placeholder for cartesian coordinates
    data_points = []
    # 1.3. Iteratively scan each organ and at it to the placeholder
    for count, file in enumerate(files):
        print(count, ": Reading ->  ", file)
        organ = open3d.read_point_cloud(file)
        organ = np.asarray(organ.points)
        data_points.append(organ)
        print("Remaining: ", len(files) - count)
    # 1.4. Save the order of our files for later
    idx = [i for i in range(len(files))]

    # 2. CNN Features --------------------------------------------------------
    # 2.1. Create empty array with dimensions n * (16*16*16)
    h, w, d = voxel_size, voxel_size, voxel_size
    X_cnn = np.zeros((data_points.shape[0], (h * w * d)), dtype=float)
    # 2.2. Iteratively voxelize each structure
    for num in range(0, data_points.shape[0]):
        # get xyz points from file
        tmp = data_points[num]
        # tmp = scale(tmp, -1, 1)
        if tmp.shape[0] != 0:
            tmp = VoxelGrid(tmp, x_y_z=[h, w, d])
            # get vector array of voxel
            tmp = tmp.vector
            if binary == True:
                tmp = tmp > 0
                tmp = tmp.astype(int)
            # flatten vector
            tmp = np.concatenate(tmp).ravel()
            # add to our list
            X_cnn[num] = tmp

    # 2. RNN Features --------------------------------------------------------
    # Create a pandas dataframe to examine and store the datafiles
    dataset_2_info = pd.DataFrame({'index': idx, 'name': files})

    if save == 'yes':
        np.save('../processed/dataset2voxels16.npy', X_cnn)
        dataset_2_info.to_csv('../../interim/dataset2labels.csv', index=True)


    return data_points, idx, X_cnn

# =============================================================================
# 2. Prepare data
# =============================================================================

if __name__ == '__main__':
    get_features(16, save="No")