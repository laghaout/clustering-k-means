# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 21:03:35 2015

@author: Amine Laghaout
"""

import numpy as np

D = 2           # Dimension of the space
N = 21          # Number of data points  
C = 2           # Number of clusters
dotSize = 60    # Size of the data markers
crossSize = 260 # Size of the centroid markers
scaling = 1     # Scaling of the randomized data points
maxIter = 10    # Maximum number of iterations of the k-means algorithm

# Data points
###############################################################################

def loadImages(N):
    
    ''' Load images as the input data to be clustered '''    
    
    from PIL import Image
    
    # Get the dimensions of the vector from the first image file    
    
    imageFile = Image.open("./images./input" + str(0) + ".png")  
    imageList = list(imageFile.getdata())
    D = len(imageList)
    V = np.empty((D, N))
    
    # Convert the images into a vector of boolean values     
    
    for currImage in np.arange(N):
    
        imageFile = Image.open("./images./input"+ str(currImage) + ".png")
        imageList = list(imageFile.getdata())
        D = len(imageList)
        boolVector = np.zeros(D)
        
        for i in np.arange(D):
            
            if sum(imageList[i]) > 256/2:        
                boolVector[i] = 1
        
        V[:, currImage] = boolVector    
    
    return (D, V)

def randData(D, N):
    
    ''' Generate distinct Gaussian clusters '''
    
    # Random data points    
    V = scaling*np.random.rand(D, N)
    
    # Random data point that are preliminarily grouped into clusters
    #V = scaling*np.concatenate((np.random.rand(D, N/2), 2+np.random.rand(D, N/2)), axis = 1) 
    
    # Data points loaded from a file
    #V = np.load('inputData.npy')
        
    return V

# Randomize input data
#V = randData(D, N)  

# Load data file image files
D, V = loadImages(N)

# Centroids
###############################################################################

def initCentroids(D, C):

    ''' Initialize the centroids '''    
    
    # Randomize the centroids
    centroids = scaling*np.random.rand(D, C)

    # Load data from a file
    #centroids = np.load('inputCentroids.npy')
    
    return centroids

centroids = initCentroids(D, C)    

# Initialization of the cluster assignments
###############################################################################

a = np.empty(N) 

# k-means algorithm
###############################################################################


for m in np.arange(maxIter):

    # For each data point...
    
    for i in np.arange(N):
        
        minDotProd = float("inf")
        
        # ... loop over all centroids...        
        
        for k in np.arange(C):
            
            dist2centroid = V[:, i] - centroids[:, k]
            dist2centroid = np.sqrt(np.dot(dist2centroid, dist2centroid))        
            
            # ... to find the closest centroid            
            
            if dist2centroid < minDotProd:
                
                minDotProd = dist2centroid   
                a[i] = k + 1
    
    # Re-center the centroids based on the new cluster assignment
    
    for k in np.arange(C):
        
        sumVectors = np.zeros(D)    
        numVectors = 0    
        
        for i in np.arange(N):
            
            if a[i] == k + 1:
                
                sumVectors += V[:, i]
                numVectors += 1
                
        centroids[:, k] = sumVectors/numVectors

# Plots
###############################################################################

# Colors

clusterColors = np.ones(C)*np.linspace(0,1,C)
aCols = np.empty(N)

for i in np.arange(N):
    aCols[i] = clusterColors[a[i]-1]

if D == 2:
    
    import matplotlib.pyplot as plt
   
    plt.scatter(centroids[0], centroids[1], s = crossSize, c = clusterColors, marker = 'x')
    plt.scatter(V[0], V[1], s = dotSize, c = aCols, marker = 'o')
    plt.show()
    
    plt.scatter(V[0], V[1], s = dotSize, marker = 'o')

elif D == 3:
    
    import matplotlib.pyplot as plt    
    
    fig = plt.figure()

    ax = fig.add_subplot(111, projection = '3d') 
    
    ax.scatter(centroids[0], centroids[1], centroids[2], s = 160, c = clusterColors, marker = 'x')
    ax.scatter(V[0], V[1], V[2], s = dotSize, c = aCols, marker = 'o')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')    
    
    plt.show()
