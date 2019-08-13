# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 15:07:05 2019

@author: z5037298
"""

# Install dependencies
import os

import pandas as pd
#!pip install open3d-python
import open3d

# Read a few organs and plot their points

# =============================================================================
# simple point cloud view
# =============================================================================
print("Load a ply point cloud, print it, and render it")

def visualise(organ):
    import numpy as np
    
    file = "../data/raw/Mesh_PtNum-1455-" + organ + ".ply"
    pcd = open3d.read_point_cloud(file)
    points = np.asarray(pcd.points)
    print(len(points))
    
    # plot point cloud 
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    	
    def plot_ply(file):
      fig = plt.figure()
      ax = fig.add_subplot(111,projection='3d')
      ax.set_ylim(-300,300)
      ax.set_xlim(-300,300)
      ax.set_zlim(-300,300)
      x=file[:,0]
      y=file[:,1]
      z=file[:,2]
      ax.scatter(x, y, z, marker='.', zdir='z')
      ax.set_xlabel('X Label')
      ax.set_ylabel('Y Label')
      ax.set_zlabel('Z Label')
      plt.show()
      
    # =============================================================================
    #   interactive view
    # =============================================================================
      # for 3D visualisation
    import plotly.graph_objs as go
    from plotly.offline import iplot, init_notebook_mode
    #!pip install plotly
    
    #Configure plotly for google colab
    def configure_plotly_browser_state():
        import IPython
        display(IPython.core.display.HTML('''
            <script src="/static/components/requirejs/require.js"></script>
            <script>
              requirejs.config({
                paths: {
                  base: '/static/base',
                  plotly: 'https://cdn.plot.ly/plotly-latest.min.js?noext',
                },
              });
            </script>
            '''))
        
    # =============================================================================
    #     
    # =============================================================================
    configure_plotly_browser_state()
    init_notebook_mode(connected=False)
    
    X=points[:,0]
    Y=points[:,1]
    Z=points[:,2]
    
    trace1 = go.Scatter3d(x=X, y=Y, z=Z, mode='markers', 
                          marker=dict(size=10, color=Z, colorscale='Viridis', opacity=0.1))
    
    data = [trace1]
    layout = go.Layout(height=500, width=600, title= "Rectum - point cloud")
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    
    # =============================================================================
    # voxel grid
    # =============================================================================
    import voxelgrid as vg
    pcdv = vg.VoxelGrid(points, x_y_z=[16,16,16])
    
    import matplotlib.pyplot 
    pcdv.plot(d=2)
    
    pcdv = pcdv.vector
    pcdv.shape
    # =============================================================================
    #   View
    # =============================================================================
    plot_ply(points)
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(pcdv, edgecolor='k')
    
# =============================================================================
# plot
# =============================================================================
organs = ['Bladder', 'BODY', 'FemoralHeadL', 'FemoralHeadR', 'PTVHD' 'Rectum']
for organ in organs:
    visualise(organ)
    
visualise('Rectum')