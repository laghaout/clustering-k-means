# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 23:45:53 2015

@author: Amine Laghaout
"""

# %%

def loadData(pathName):

    from numpy import matrix

    fileList = getFileList(pathName)
    
    dataMatrix = [vectorizeFile(file) for file in fileList]

    # TO-DO: Check that all data have the same dimensions

    return matrix(dataMatrix) 

def getFileList(pathName):

    ''' Return the list of files at `pathName` '''
    
    # TO-DO: Check that a wildcard is part of the path name

    from glob import glob
   
    fileList = glob(pathName)

    return fileList

# %%

def vectorizeFile(filePath):
    
    ''' Convert the file at `filePath` into a matrix '''

    import os.path
    
    fileExtension = os.path.splitext(filePath)[1][1:].lower()

    imageFileExt = ['png', 'jpg', 'jpeg', 'bmp', 'gif']
    
    if fileExtension in imageFileExt:
        
        from PIL import Image      
        
        dataVector = list(Image.open(filePath).convert('L').getdata())
        
    else:
        
        # TO-DO: Handle different input files
        print('ERROR')
      
    return dataVector

