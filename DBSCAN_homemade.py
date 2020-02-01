import os
import pandas as pd
from numpy import sqrt
import numpy as np
from copy import deepcopy
import csv
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import pylab as pl
from itertools import cycle


## Import raw data


os.chdir('/Users/albert/Desktop/DBSCAN')

with open('all_locs_mar19_good_in32fb.16fb.tsv') as file:
    
    imported_xyz = csv.reader(file, delimiter='\t')
  
    final_list = []  
    
    for row in imported_xyz:
        
        temp = []
        
        for entry in row[0].split(' '):
            
            if entry == row[0].split(' ')[-1]:
                
                temp.append(int(entry)*0.24)
                
            else:

                temp.append(int(entry)*0.256)
            
        final_list.append(temp)


xyz_df = pd.DataFrame(final_list, columns = ['x', 'y', 'z'])
xyz_df['cluster'] = 0






## Get dataframe of distances:


all_dist_of_points = []
for row in xyz_df.index:
    
    x_current = xyz_df.at[row,'x']
    y_current = xyz_df.at[row,'y']

    all_dist_of_points.append([ sqrt((x_current-xyz_df.at[row2,'x'])**2 + (y_current-xyz_df.at[row2,'y'])**2) for row2 in xyz_df.index ])
            
    print('Found neighbour distances of point {}'.format(row))
    
    
dist_df = pd.DataFrame(all_dist_of_points)



## Using values of epsilon and min_pts, get lists of core_points and non_core_points:

epsilon = 150
min_pts = 75


core_points = []
noncore_points = []




for node_name in dist_df.index:
    
    # Get list of distances (entries of row) with every other point into a list:
    list_of_distances_this_row = list(dist_df.iloc[node_name])
    
    
    # Make a new list of the points who's distance is less than epsilon:    
    dr_points = [list_of_distances_this_row.index(x) for x in list_of_distances_this_row if x<= epsilon]    

    
    
    # If the legnth of this list is greater or equal to min_pts number, then this popint is  a core point:
    if len(dr_points) >= min_pts:
        
        core_points.append([node_name, dr_points])
    
    else:
        
        noncore_points.append([node_name, dr_points])
            
            

list_of_core_points = [x[0] for x in core_points]
    


core_adj_mat = pd.DataFrame(np.zeros(shape=(len(core_points),len(core_points))), dtype=int, index=list_of_core_points, columns=list_of_core_points)



for entry in core_points:
    
    for partner in entry[1]:
        
        if partner in list_of_core_points:
            
            core_adj_mat.at[entry[0], partner] = int(1)
            core_adj_mat.at[partner, entry[0]] = int(1)
 
 
 
 
## Getting connected components (in an inefficient way, could be replaced by package later)

connected_components = []


for core_point in core_adj_mat.index:
    
    membership = 'no'
    
    core_point_neighbours = []
    
    for entry in list(core_adj_mat.loc[core_point]):
        
        if entry == 1:
            core_point_neighbours.append(list(core_adj_mat.columns)[list(core_adj_mat.loc[core_point]).index(entry)])
            
    
    list(core_adj_mat.loc[core_point])
    
    for cc in connected_components:
        
        if membership == 'no':
        
            for node in cc:
                
                if membership == 'no':
                
                    if node in core_point_neighbours:
                        
                        cc.append(core_point)
                        
                        membership = 'yes'
    
    if membership == 'no':
        
        connected_components.append([core_point])
        
        

for cc in connected_components:
    
    cc = list(set(cc))
    


connected_components_with_non_core = deepcopy(connected_components)


    
for sublist in noncore_points:
    
    neighbours = sublist[1]   
    
    assigned_to_cluster = 'no'
    
    for neighbour in neighbours:
        
        if assigned_to_cluster == 'no':
        
            for con_com in connected_components:
                
                if assigned_to_cluster == 'no':
                
                    if neighbour in con_com:
                        
                        connected_components_with_non_core[connected_components.index(con_com)].append(sublist[0])
        
                        assigned_to_cluster = 'yes'
                
                
                

cluster_num = []

for connected_component in connected_components_with_non_core:
    
    cc_num = connected_components_with_non_core.index(connected_component)
    
    for node in connected_component:
        
        xyz_df.at[node,'cluster'] = cc_num+1
        
    cluster_num.append(cc_num)




## Save dataframe
os.mkdir('xyz_df_e'+str(epsilon)+'_minpts'+str(min_pts))
os.chdir('xyz_df_e'+str(epsilon)+'_minpts'+str(min_pts))
xyz_df.to_csv('xyz_df_e'+str(epsilon)+'_minpts'+str(min_pts)+'.csv', index=False)



## Visual Representation 


import matplotlib

x = list(xyz_df['x'])
y = list(xyz_df['y'])
label = list(xyz_df['cluster'])
colors = list(matplotlib.colors.cnames.keys())

fig = plt.figure(figsize=(8,8))
plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))

plt.show()



                        


