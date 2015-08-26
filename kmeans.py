# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:03:35 2015

@author: Amine Laghaout
"""

from loadData import loadData
from sklearn import cluster

C = 3                           # Number of clusters
pathName = './images/*'         # Path name to the data files
data = loadData(pathName)

# TO-DO: Normalize the data

k_means = cluster.KMeans(n_clusters = C)
k_means.fit(data)
print(k_means.labels_) 
