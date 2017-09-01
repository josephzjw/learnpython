# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 19:22:34 2017

@author: z81022682
"""

__author__ = 'Xue Er Shi Xi Ji'

import json
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

chessArray = {'101':1, '201':2, '302':4, '301':2, 
                  '405':1, '402':4, '401':2, '404':8, '403':4, 
                  '508':8, '502':4, '504':4, '510':8, '511':4, '501':2, 
                  '506':8, '505':4, '509':4, '512':8, '503':1, '507':8}
offsetArray = {'101':0, '201':20*20*1, '302':20*20*(1+2), '301':20*20*(1+2+4), 
                  '405':20*20*(1+2+4+2), '402':20*20*(1+2+4+2+1), '401':20*20*(1+2+4+2+1+4), '404':20*20*(1+2+4+2+1+4+2), '403':20*20*(1+2+4+2+1+4+2+8), 
                  '508':20*20*(1+2+4+2+1+4+2+8+4), '502':20*20*(1+2+4+2+1+4+2+8+4+8), '504':20*20*(1+2+4+2+1+4+2+8+4+8+4), 
                  '510':20*20*(1+2+4+2+1+4+2+8+4+8+4+4), '511':20*20*(1+2+4+2+1+4+2+8+4+8+4+4+8), '501':20*20*(1+2+4+2+1+4+2+8+4+8+4+4+8+4), 
                  '506':20*20*(1+2+4+2+1+4+2+8+4+8+4+4+8+4+2), '505':20*20*(1+2+4+2+1+4+2+8+4+8+4+4+8+4+2+8), 
                  '509':20*20*(1+2+4+2+1+4+2+8+4+8+4+4+8+4+2+8+4), '512':20*20*(1+2+4+2+1+4+2+8+4+8+4+4+8+4+2+8+4+4), 
                  '503':20*20*(1+2+4+2+1+4+2+8+4+8+4+4+8+4+2+8+4+4+8), '507':20*20*(1+2+4+2+1+4+2+8+4+8+4+4+8+4+2+8+4+4+8+1)}
ROW=20
COLUMN=20
PID1 = 1;
PID2 = 2;
PID3 = 4;
PID4 = 8;
PIDUS = 1;
PIDFD = 2;
PIDEMY = 4;
status_count = ROW * COLUMN * (1 + 2 + 4 + 2 + 1 + 4 + 2 + 8 + 4 + 8 + 4 + 4 + 8 + 4 + 2 + 8 + 4 + 4 + 8 + 1 + 8)


################### myPlot(myCurrentChessState[:,:,step]) #####################
def myPlot(H):
    fig = plt.figure(figsize=(8, 4))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(H)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()


#################### myplt(label2axis(myAxisLabel[step])) #####################
def myplt(data):
    numOfScatter = len(data)
    ix = []
    iy = []
    for i in range(numOfScatter):
        ix.append(data[i][0])
        iy.append(data[i][1])
    plt.scatter(iy,ix)
    plt.xlim([-5,24])
    plt.ylim([-5,24])
    plt.gca().invert_yaxis()

#print(os.path.isfile(fileName))
###############################################################################   


############################## map from axis to label #########################
def axis2label(axisList):
    myMap = []
    # 101: 400
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j]])

# 201: 800
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j+1]])
            myMap.append([[i,j],[i+1,j]])
    
# 301: 800
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j-1],[i,j],[i,j+1]])
            myMap.append([[i-1,j],[i,j],[i+1,j]])

# 302: 1600
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j-1],[i,j],[i+1,j]])
            myMap.append([[i,j-1],[i,j],[i-1,j]])
            myMap.append([[i,j],[i-1,j],[i,j+1]])
            myMap.append([[i,j],[i+1,j],[i,j+1]])

# 401: 800
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j-1],[i,j],[i,j+1],[i,j+2]])
            myMap.append([[i-1,j],[i,j],[i+1,j],[i+2,j]])

# 402: 4*400
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j-1],[i,j],[i,j+1],[i+1,j]])
            myMap.append([[i-1,j],[i,j],[i+1,j],[i,j+1]])
            myMap.append([[i,j-1],[i,j],[i,j+1],[i-1,j]])
            myMap.append([[i,j-1],[i,j],[i-1,j],[i+1,j]])

# 403: 4*400
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i+1,j],[i,j+1],[i+1,j-1]])
            myMap.append([[i,j],[i,j-1],[i-1,j-1],[i+1,j]])
            myMap.append([[i,j],[i,j-1],[i+1,j],[i+1,j+1]])
            myMap.append([[i,j],[i-1,j],[i,j-1],[i+1,j-1]])
        
# 404: 8*400
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j+1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i+1,j+1]])
            myMap.append([[i,j],[i,j-1],[i+1,j-1],[i,j+1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i-1,j-1]])
            myMap.append([[i,j],[i,j-1],[i-1,j-1],[i,j+1]])
            myMap.append([[i,j],[i+1,j],[i-1,j],[i-1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j+1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i+1,j-1]])

# 405: 1
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j+1],[i+1,j],[i+1,j+1]])
        
        
# 501: 2*400
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i,j-2],[i,j+1],[i,j+2]])
            myMap.append([[i,j],[i-1,j],[i-2,j],[i+1,j],[i+2,j]])

# 502: 4
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i-1,j],[i-2,j],[i,j-1],[i,j+1]])
            myMap.append([[i,j],[i,j+1],[i,j+2],[i-1,j],[i+1,j]])
            myMap.append([[i,j],[i+1,j],[i+2,j],[i,j-1],[i,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i-1,j],[i+1,j]])

# 503: 1
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j],[i+1,j]])

## 504: 4
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j+1],[i,j+2],[i-1,j],[i-2,j]])
            myMap.append([[i,j],[i+1,j],[i+2,j],[i,j+1],[i,j+2]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i+1,j],[i+2,j]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i-1,j],[i-2,j]])
        
## 505: 4
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i+1,j-1],[i-1,j],[i-1,j+1]])
            myMap.append([[i,j],[i-1,j],[i-1,j-1],[i,j+1],[i+1,j+1]])
            myMap.append([[i,j],[i+1,j],[i+1,j-1],[i,j+1],[i-1,j+1]])
            myMap.append([[i,j],[i,j-1],[i-1,j-1],[i+1,j],[i+1,j+1]])
        

## 506: 8
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i-1,j],[i+1,j],[i,j+1],[i+1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j-1],[i+1,j]])
            myMap.append([[i,j],[i-1,j],[i-1,j-1],[i,j-1],[i+1,j]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j],[i-1,j+1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i,j-1],[i+1,j-1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j],[i-1,j-1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i,j+1],[i-1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j],[i+1,j+1]])

## 507: 8
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i,j+1],[i,j+2],[i-1,j]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i+2,j],[i,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i,j+1],[i+1,j]])
            myMap.append([[i,j],[i+1,j],[i-1,j],[i-2,j],[i,j-1]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i-1,j],[i,j+1]])
            myMap.append([[i,j],[i+1,j],[i-1,j],[i-2,j],[i,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i,j+2],[i+1,j]])
            myMap.append([[i,j],[i,j-1],[i-1,j],[i+1,j],[i+2,j]])

## 508: 8
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i-1,j],[i,j+1],[i,j+2],[i,j+3]])
            myMap.append([[i,j],[i,j+1],[i+1,j],[i+2,j],[i+3,j]])
            myMap.append([[i,j],[i+1,j],[i,j-1],[i,j-2],[i,j-3]])
            myMap.append([[i,j],[i,j-1],[i-1,j],[i-2,j],[i-3,j]])
            myMap.append([[i,j],[i-1,j],[i,j-1],[i,j-2],[i,j-3]])
            myMap.append([[i,j],[i,j+1],[i-1,j],[i-2,j],[i-3,j]])
            myMap.append([[i,j],[i+1,j],[i,j+1],[i,j+2],[i,j+3]])
            myMap.append([[i,j],[i,j-1],[i+1,j],[i+2,j],[i+3,j]])

## 509: 4
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i+1,j],[i-1,j],[i-1,j+1],[i+1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j-1],[i+1,j+1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i-1,j-1],[i+1,j-1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j-1],[i-1,j+1]])


## 510: 8
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i+1,j],[i+1,j-1],[i,j+1],[i,j+2]])
            myMap.append([[i,j],[i,j-1],[i-1,j-1],[i+1,j],[i+2,j]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i-1,j],[i-1,j+1]])
            myMap.append([[i,j],[i,j+1],[i+1,j+1],[i-1,j],[i-2,j]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i+1,j],[i+1,j+1]])
            myMap.append([[i,j],[i,j-1],[i+1,j-1],[i-1,j],[i-2,j]])
            myMap.append([[i,j],[i,j+1],[i,j+2],[i-1,j],[i-1,j-1]])
            myMap.append([[i,j],[i+1,j],[i+2,j],[i,j+1],[i-1,j+1]])

## 511: 4
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j-1],[i-1,j+1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i-1,j-1],[i+1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j-1],[i+1,j+1]])
            myMap.append([[i,j],[i+1,j],[i-1,j],[i+1,j-1],[i-1,j+1]])

## 512: 8
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i+1,j],[i-1,j],[i-1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j],[i+1,j+1]])
            myMap.append([[i,j],[i+1,j],[i-1,j],[i,j+1],[i+1,j-1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j],[i-1,j-1]])
            myMap.append([[i,j],[i,j+1],[i+1,j],[i-1,j],[i-1,j-1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j],[i-1,j+1]])
            myMap.append([[i,j],[i+1,j],[i-1,j],[i,j-1],[i+1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j],[i+1,j-1]])
            
    myMapSorted = []
    for i in range(36400):
        myMapSorted.append(sorted(myMap[i]))

    return myMapSorted.index(axisList)

################# map from axis to label end   ###############################
##############################################################################


###############################################################################
#####################   map from label to axis ################################
def label2axis(mylabel):
    myMap = []
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j]])

# 201: 800
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j+1]])
            myMap.append([[i,j],[i+1,j]])
    
# 301: 800
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j-1],[i,j],[i,j+1]])
            myMap.append([[i-1,j],[i,j],[i+1,j]])

# 302: 1600
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j-1],[i,j],[i+1,j]])
            myMap.append([[i,j-1],[i,j],[i-1,j]])
            myMap.append([[i,j],[i-1,j],[i,j+1]])
            myMap.append([[i,j],[i+1,j],[i,j+1]])

# 401: 800
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j-1],[i,j],[i,j+1],[i,j+2]])
            myMap.append([[i-1,j],[i,j],[i+1,j],[i+2,j]])

# 402: 4*400
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j-1],[i,j],[i,j+1],[i+1,j]])
            myMap.append([[i-1,j],[i,j],[i+1,j],[i,j+1]])
            myMap.append([[i,j-1],[i,j],[i,j+1],[i-1,j]])
            myMap.append([[i,j-1],[i,j],[i-1,j],[i+1,j]])

# 403: 4*400
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i+1,j],[i,j+1],[i+1,j-1]])
            myMap.append([[i,j],[i,j-1],[i-1,j-1],[i+1,j]])
            myMap.append([[i,j],[i,j-1],[i+1,j],[i+1,j+1]])
            myMap.append([[i,j],[i-1,j],[i,j-1],[i+1,j-1]])
        
# 404: 8*400
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j+1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i+1,j+1]])
            myMap.append([[i,j],[i,j-1],[i+1,j-1],[i,j+1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i-1,j-1]])
            myMap.append([[i,j],[i,j-1],[i-1,j-1],[i,j+1]])
            myMap.append([[i,j],[i+1,j],[i-1,j],[i-1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j+1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i+1,j-1]])

# 405: 1
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j+1],[i+1,j],[i+1,j+1]])
        
        
# 501: 2*400
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i,j-2],[i,j+1],[i,j+2]])
            myMap.append([[i,j],[i-1,j],[i-2,j],[i+1,j],[i+2,j]])

# 502: 4
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i-1,j],[i-2,j],[i,j-1],[i,j+1]])
            myMap.append([[i,j],[i,j+1],[i,j+2],[i-1,j],[i+1,j]])
            myMap.append([[i,j],[i+1,j],[i+2,j],[i,j-1],[i,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i-1,j],[i+1,j]])

# 503: 1
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j],[i+1,j]])

## 504: 4
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j+1],[i,j+2],[i-1,j],[i-2,j]])
            myMap.append([[i,j],[i+1,j],[i+2,j],[i,j+1],[i,j+2]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i+1,j],[i+2,j]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i-1,j],[i-2,j]])
        
## 505: 4
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i+1,j-1],[i-1,j],[i-1,j+1]])
            myMap.append([[i,j],[i-1,j],[i-1,j-1],[i,j+1],[i+1,j+1]])
            myMap.append([[i,j],[i+1,j],[i+1,j-1],[i,j+1],[i-1,j+1]])
            myMap.append([[i,j],[i,j-1],[i-1,j-1],[i+1,j],[i+1,j+1]])
        

## 506: 8
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i-1,j],[i+1,j],[i,j+1],[i+1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j-1],[i+1,j]])
            myMap.append([[i,j],[i-1,j],[i-1,j-1],[i,j-1],[i+1,j]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j],[i-1,j+1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i,j-1],[i+1,j-1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j],[i-1,j-1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i,j+1],[i-1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j],[i+1,j+1]])

## 507: 8
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i,j+1],[i,j+2],[i-1,j]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i+2,j],[i,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i,j+1],[i+1,j]])
            myMap.append([[i,j],[i+1,j],[i-1,j],[i-2,j],[i,j-1]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i-1,j],[i,j+1]])
            myMap.append([[i,j],[i+1,j],[i-1,j],[i-2,j],[i,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i,j+2],[i+1,j]])
            myMap.append([[i,j],[i,j-1],[i-1,j],[i+1,j],[i+2,j]])

## 508: 8
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i-1,j],[i,j+1],[i,j+2],[i,j+3]])
            myMap.append([[i,j],[i,j+1],[i-1,j],[i-2,j],[i-3,j]])
            myMap.append([[i,j],[i+1,j],[i,j-1],[i,j-2],[i,j-3]])
            myMap.append([[i,j],[i,j-1],[i-1,j],[i-2,j],[i-3,j]])
            myMap.append([[i,j],[i-1,j],[i,j-1],[i,j-2],[i,j-3]])
            myMap.append([[i,j],[i,j+1],[i-1,j],[i-2,j],[i-3,j]])
            myMap.append([[i,j],[i+1,j],[i,j+1],[i,j+2],[i,j+3]])
            myMap.append([[i,j],[i,j-1],[i+1,j],[i+2,j],[i+3,j]])

## 509: 4
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i+1,j],[i-1,j],[i-1,j+1],[i+1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j-1],[i+1,j+1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i-1,j-1],[i+1,j-1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j-1],[i-1,j+1]])


## 510: 8
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i+1,j],[i+1,j-1],[i,j+1],[i,j+2]])
            myMap.append([[i,j],[i,j-1],[i-1,j-1],[i+1,j],[i+2,j]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i-1,j],[i-1,j+1]])
            myMap.append([[i,j],[i,j+1],[i+1,j+1],[i-1,j],[i-2,j]])
            myMap.append([[i,j],[i,j-1],[i,j-2],[i+1,j],[i+1,j+1]])
            myMap.append([[i,j],[i,j-1],[i+1,j-1],[i-1,j],[i-2,j]])
            myMap.append([[i,j],[i,j+1],[i,j+2],[i-1,j],[i-1,j-1]])
            myMap.append([[i,j],[i+1,j],[i+2,j],[i,j+1],[i-1,j+1]])

## 511: 4
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j-1],[i-1,j+1]])
            myMap.append([[i,j],[i-1,j],[i+1,j],[i-1,j-1],[i+1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j-1],[i+1,j+1]])
            myMap.append([[i,j],[i+1,j],[i-1,j],[i+1,j-1],[i-1,j+1]])

## 512: 8
    for i in range(20):
        for j in range(20):
            myMap.append([[i,j],[i,j-1],[i+1,j],[i-1,j],[i-1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j],[i+1,j+1]])
            myMap.append([[i,j],[i+1,j],[i-1,j],[i,j+1],[i+1,j-1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j],[i-1,j-1]])
            myMap.append([[i,j],[i,j+1],[i+1,j],[i-1,j],[i-1,j-1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i+1,j],[i-1,j+1]])
            myMap.append([[i,j],[i+1,j],[i-1,j],[i,j-1],[i+1,j+1]])
            myMap.append([[i,j],[i,j-1],[i,j+1],[i-1,j],[i+1,j-1]])
    
    myMapSorted = []
    for i in range(36400):
        myMapSorted.append(sorted(myMap[i]))
    return myMapSorted[mylabel]

############ map from label to axis end #######################################
###############################################################################      


##############################################################################
######### usage: myplt2(myCurrentChessState[:,:,1],myAxisLabel[3]) ###########
def myplt2(chessState,label):
    chessIndex = np.argwhere(chessState)
    labelAxis = label2axis(label)
    ix = []
    iy = []
    for i in range(len(chessIndex)):
        ix.append(chessIndex[i][0])
        iy.append(chessIndex[i][1])
        
    for i in range(len(labelAxis)):
        ix.append(labelAxis[i][0])
        iy.append(labelAxis[i][1])
        
    plt.scatter(iy,ix)
    plt.xlim([0,19])
    plt.ylim([0,19])
    plt.gca().invert_yaxis()
    
################### myplt2 end ################################################
    
    
################### chess board rotate ########################################
### input:  currentChessBoard
###         pid: player id
### output: newChessBoard
def myChessBoardRotate(currentChessBoard,pid):
    chessLoc = np.argwhere(currentChessBoard)
    rotateChessBoard = np.zeros((20,20))
    if pid == 1:
        rotateChessBoard = currentChessBoard
    elif pid == 2:
        for i in range(len(chessLoc)):
            rotateChessBoard[19-chessLoc[i][1]][chessLoc[i][0]] = currentChessBoard[chessLoc[i][0]][chessLoc[i][1]]
    elif pid == 3:
        for i in range(len(chessLoc)):
            rotateChessBoard[19-chessLoc[i][0]][19-chessLoc[i][1]] = currentChessBoard[chessLoc[i][0]][chessLoc[i][1]]
    elif pid == 4:
        for i in range(len(chessLoc)):
            rotateChessBoard[chessLoc[i][1]][19-chessLoc[i][0]] = currentChessBoard[chessLoc[i][0]][chessLoc[i][1]]
        
    return rotateChessBoard
#################### chess board rotation end #################################

###################### chess rotation #########################################
##### input: label
#####        pid: player id
##### output: rotated label            
def myChessRotate(label,pid):
    axisSet = label2axis(label)
    rotatedLabel = -2
    rotatedAxis = []
    if pid == 1:
        rotatedLabel = label
    elif pid == 2:
        for i in range(len(axisSet)):
            rotatedAxis.append([ 19-axisSet[i][1], axisSet[i][0] ])
    elif pid == 3:
        for i in range(len(axisSet)):
            rotatedAxis.append([ 19-axisSet[i][0], 19-axisSet[i][1] ])
    elif pid == 4:
        for i in range(len(axisSet)):
            rotatedAxis.append([ axisSet[i][1], 19-axisSet[i][0] ])
    
    if pid != 1:
        rotatedAxisSorted = sorted(rotatedAxis)
        rotatedLabel = axis2label(rotatedAxisSorted)
    return rotatedLabel
    
###################### chess rotation end #####################################

###################### data extraction   ######################################
### input: the file name
### output: the chessboard state: (20*20*numOfHand)
###         myLabelStep: (1*numOfHand)

def myDataProcess(fileName):
    with open(fileName, 'r') as f:
        mydata = f.read()
        mydata_new = mydata[10:-1]
    
    data = json.loads(mydata_new)

    numOfHand = len(data)-2 # the total steps of the chess game
    myCurrentChessState = np.zeros((20,20,numOfHand+1),dtype=np.int) # what does the current chessboard look like
    
    myAxisLabel = []
    for i in range(numOfHand): # the i-th hand
        if i != 0:
            myCurrentChessState[:,:,i+1] = myCurrentChessState[:,:,i]
        if bool(data[i+1]['msg_data']['chessman'].get('squareness')): # this player can play
            numOfSquareness = len(data[i+1]['msg_data']['chessman']['squareness'])
            myTmpAxis = []
            if numOfSquareness >1:
                for j in range(numOfSquareness):
                    xid = data[i+1]['msg_data']['chessman']['squareness'][j]['x'] # x_axis
                    yid = data[i+1]['msg_data']['chessman']['squareness'][j]['y'] # y_axis
                    zid = data[i+1]['msg_data']['player_id'] # player_id
                    myCurrentChessState[xid,yid,i+1] = zid
                    myTmpAxis.append([xid,yid])
            else:
                myTmpAxis = [[data[i+1]['msg_data']['chessman']['squareness'][0]['x'],data[i+1]['msg_data']['chessman']['squareness'][0]['y']]]

            myTmpAxisSorted = sorted(myTmpAxis)
            myAxisLabel.append(axis2label(myTmpAxisSorted))
        else:
            myAxisLabel.append(-1)
            
    return myCurrentChessState, myAxisLabel
##################### chess board rotate end ##################################
    
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
  

def overlap_check(chessboard, row, column, chess, orientation):
#This function will check whether the next step out of the chessboard, or there were already a chess
#Return 0 means the step is invalid, 1 means the stpe is valid
#(row, column) - the position for the anchor point
#chess - the index of chess
#orientation - the rotation of the chess
    if chessboard[row][column] == 0:
        #the red circle block valid
        if chess == '101':
            if orientation == 0:
                return 1
            else:
                return -1
        elif chess == '201':
            if orientation == 0:
                if column < COLUMN - 1:
                    if chessboard[row][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if row < ROW - 1:
                    if chessboard[row + 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '302':
            if orientation == 0:
                if row < ROW - 1 and column > 0:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if row > 0 and column > 0:
                    if chessboard[row - 1][column] == 0 and chessboard[row][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if row > 0 and column < COLUMN - 1:
                    if chessboard[row - 1][column] == 0 and chessboard[row][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if row < ROW - 1 and column < COLUMN - 1:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '301':
            if orientation == 0:
                 if column < COLUMN - 1 and column > 0:
                    if chessboard[row][column - 1] == 0 and chessboard[row][column + 1] == 0:
                        return 1
                    else:
                        return 0
                 else:
                    return 0
            elif orientation == 1:
                if row < ROW - 1 and row > 0:
                    if chessboard[row + 1][column] == 0 and chessboard[row - 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '405':
            if orientation == 0:
                if row < ROW - 1 and column < COLUMN - 1:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column + 1] == 0 and chessboard[row + 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '402':
            if orientation == 0:
                if column > 0 and column < COLUMN - 1 and row > 0:
                    if chessboard[row - 1][column] == 0 and chessboard[row][column + 1] == 0 and chessboard[row][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if column < COLUMN - 1 and row > 0 and row < ROW - 1:
                    if chessboard[row][column + 1] == 0 and chessboard[row - 1][column] == 0 and chessboard[row + 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if column > 0 and column < COLUMN - 1 and row < ROW - 1:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column + 1] == 0 and chessboard[row][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if column > 0 and row > 0 and row < ROW - 1:
                    if chessboard[row][column - 1] == 0 and chessboard[row - 1][column] == 0 and chessboard[row + 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '401':
            if orientation == 0:
                if column < COLUMN - 2 and column > 0:
                    if chessboard[row][column - 1] == 0 and chessboard[row][column + 1] == 0 and chessboard[row][column + 2] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if row < ROW - 2 and row > 0:
                    if chessboard[row + 2][column] == 0 and chessboard[row + 1][column] == 0 and chessboard[row - 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '404':
            if orientation == 0:
                if column > 0 and column < COLUMN - 1 and row > 0:
                    if chessboard[row - 1][column + 1] == 0 and chessboard[row][column + 1] == 0 and chessboard[row][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if column < COLUMN - 1 and row > 0 and row < ROW - 1:
                    if chessboard[row + 1][column + 1] == 0 and chessboard[row - 1][column] == 0 and chessboard[row + 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if column > 0 and column < COLUMN - 1 and row < ROW - 1:
                    if chessboard[row + 1][column - 1] == 0 and chessboard[row][column + 1] == 0 and chessboard[row][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if column > 0 and row > 0 and row < ROW - 1:
                    if chessboard[row - 1][column - 1] == 0 and chessboard[row - 1][column] == 0 and chessboard[row + 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 4:
                if column > 0 and column < COLUMN - 1 and row > 0:
                    if chessboard[row - 1][column - 1] == 0 and chessboard[row][column + 1] == 0 and chessboard[row][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 5:
                if column < COLUMN - 1 and row > 0 and row < ROW - 1:
                    if chessboard[row - 1][column + 1] == 0 and chessboard[row - 1][column] == 0 and chessboard[row + 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 6:
                if column > 0 and column < COLUMN - 1 and row < ROW - 1:
                    if chessboard[row + 1][column + 1] == 0 and chessboard[row][column + 1] == 0 and chessboard[row][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 7:
                if column > 0 and row > 0 and row < ROW - 1:
                    if chessboard[row + 1][column - 1] == 0 and chessboard[row - 1][column] == 0 and chessboard[row + 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '403':
            if orientation == 0:
                if column > 0 and column < COLUMN - 1 and row < ROW - 1:
                    if chessboard[row + 1][column - 1] == 0 and chessboard[row + 1][column] == 0 and chessboard[row][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if column > 0 and row > 0 and row < ROW - 1:
                    if chessboard[row - 1][column - 1] == 0 and chessboard[row][column - 1] == 0 and chessboard[row + 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if column > 0 and column < COLUMN - 1 and row < ROW - 1:
                    if chessboard[row][column - 1] == 0 and chessboard[row + 1][column] == 0 and chessboard[row + 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if column > 0 and row > 0 and row < ROW - 1:
                    if chessboard[row - 1][column] == 0 and chessboard[row][column - 1] == 0 and chessboard[row + 1][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '508':
            if orientation == 0:
                if row > 0 and column < COLUMN - 3:
                    if chessboard[row - 1][column] == 0 and chessboard[row][column + 1] == 0 and \
                        chessboard[row][column + 2] == 0 and chessboard[row][column + 3] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if row < ROW - 3 and column < COLUMN - 1:
                    if chessboard[row][column + 1] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row + 2][column] == 0 and chessboard[row + 3][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if row < ROW - 1 and column > 2:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row][column - 2] == 0 and chessboard[row][column - 3] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if row > 2 and column > 0:
                    if chessboard[row][column - 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row - 2][column] == 0 and chessboard[row - 3][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 4:
                if row > 0 and column > 2:
                    if chessboard[row - 1][column] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row][column - 2] == 0 and chessboard[row][column - 3] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 5:
                if row > 2 and column < COLUMN - 1:
                    if chessboard[row][column + 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row - 2][column] == 0 and chessboard[row - 3][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 6:
                if row < ROW - 1 and column < COLUMN - 3:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column + 1] == 0 and \
                        chessboard[row][column + 2] == 0 and chessboard[row][column + 3] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 7:
                if row < ROW - 3 and column > 0:
                    if chessboard[row][column - 1] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row + 2][column] == 0 and chessboard[row + 3][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '502':
            if orientation == 0:
                if row > 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row][column - 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row - 2][column] == 0 and chessboard[row][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if row > 0 and row < ROW - 1 and column < COLUMN - 2:
                    if chessboard[row - 1][column] == 0 and chessboard[row][column + 1] == 0 and \
                        chessboard[row][column + 2] == 0 and chessboard[row + 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if row < ROW - 2 and column > 0 and column < COLUMN - 1:
                    if chessboard[row][column - 1] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row + 2][column] == 0 and chessboard[row][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if row > 0 and row < ROW - 1 and column > 1:
                    if chessboard[row - 1][column] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row][column - 2] == 0 and chessboard[row + 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '504':
            if orientation == 0:
                if row > 1 and column < COLUMN - 2:
                    if chessboard[row - 2][column] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row][column + 2] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if row < ROW - 2 and column < COLUMN - 2:
                    if chessboard[row][column + 2] == 0 and chessboard[row][column + 1] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row + 2][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if row < ROW - 2 and column > 1:
                    if chessboard[row + 2][column] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row][column - 1] == 0 and chessboard[row][column - 2] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if row > 1 and column > 1:
                    if chessboard[row][column - 2] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row - 1][column] == 0 and chessboard[row - 2][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '510':
            if orientation == 0:
                if row < ROW - 1 and column > 0 and column < COLUMN - 2:
                    if chessboard[row + 1][column - 1] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row][column + 2] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if row > 0 and row < ROW - 2 and column > 0:
                    if chessboard[row - 1][column - 1] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row + 2][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if row > 0 and column > 1 and column < COLUMN - 1:
                    if chessboard[row - 1][column + 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row][column - 1] == 0 and chessboard[row][column - 2] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if row > 1 and row < ROW - 1 and column < COLUMN - 1:
                    if chessboard[row + 1][column + 1] == 0 and chessboard[row][column + 1] == 0 and \
                        chessboard[row - 1][column] == 0 and chessboard[row - 2][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 4:
                if row < ROW - 1 and column > 1 and column < COLUMN - 1:
                    if chessboard[row + 1][column + 1] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row][column - 1] == 0 and chessboard[row][column - 2] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 5:
                if row > 1 and row < ROW - 1 and column > 0:
                    if chessboard[row + 1][column - 1] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row - 1][column] == 0 and chessboard[row - 2][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 6:
                if row > 0 and column > 0 and column < COLUMN - 2:
                    if chessboard[row - 1][column - 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row][column + 2] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 7:
                if row > 0 and row < ROW - 2 and column < COLUMN - 1:
                    if chessboard[row - 1][column + 1] == 0 and chessboard[row][column + 1] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row + 2][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '511':
            if orientation == 0:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row + 1][column - 1] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row - 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row - 1][column - 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row + 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row - 1][column - 1] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row + 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row - 1][column + 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row + 1][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '501':
            if orientation == 0:
                 if column < COLUMN - 2 and column > 1:
                    if chessboard[row][column - 2] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row][column + 2] == 0:
                        return 1
                    else:
                        return 0
                 else:
                    return 0
            elif orientation == 1:
                if row < ROW - 2 and row > 1:
                    if chessboard[row - 2][column] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row + 2][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '506':
            if orientation == 0:
                if column < COLUMN - 1 and row > 0 and row < ROW - 1:
                    if chessboard[row - 1][column] == 0 and chessboard[row][column + 1] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row + 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if column > 0 and column < COLUMN - 1 and row < ROW - 1:
                    if chessboard[row][column + 1] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row][column - 1] == 0 and chessboard[row + 1][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if column > 0 and row > 0 and row < ROW - 1:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row - 1][column] == 0 and chessboard[row - 1][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if column > 0 and column < COLUMN - 1 and row > 0:
                    if chessboard[row][column - 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row - 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 4:
                if column > 0 and row > 0 and row < ROW - 1:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row - 1][column] == 0 and chessboard[row + 1][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 5:
                if column > 0 and column < COLUMN - 1 and row > 0:
                    if chessboard[row][column - 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row - 1][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 6:
                if column < COLUMN - 1 and row > 0 and row < ROW - 1:
                    if chessboard[row - 1][column] == 0 and chessboard[row][column + 1] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row - 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 7:
                if column > 0 and column < COLUMN - 1 and row < ROW - 1:
                    if chessboard[row][column + 1] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row][column - 1] == 0 and chessboard[row + 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '505':
            if orientation == 0:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row + 1][column - 1] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row - 1][column] == 0 and chessboard[row - 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row - 1][column - 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row + 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row + 1][column - 1] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row - 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row - 1][column - 1] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row + 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '509':
            if orientation == 0:
                if column < COLUMN - 1 and row > 0 and row < ROW - 1:
                    if chessboard[row - 1][column] == 0 and chessboard[row - 1][column + 1] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row + 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if column > 0 and column < COLUMN - 1 and row < ROW - 1:
                    if chessboard[row][column + 1] == 0 and chessboard[row + 1][column + 1] == 0 and \
                        chessboard[row][column - 1] == 0 and chessboard[row + 1][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if column > 0 and row > 0 and row < ROW - 1:
                    if chessboard[row + 1][column] == 0 and chessboard[row + 1][column - 1] == 0 and \
                        chessboard[row - 1][column] == 0 and chessboard[row - 1][column - 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if column > 0 and column < COLUMN - 1 and row > 0:
                    if chessboard[row][column - 1] == 0 and chessboard[row - 1][column - 1] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row - 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '512':
            if orientation == 0:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row - 1][column] == 0 and chessboard[row - 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row][column - 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row + 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row + 1][column - 1] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row - 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row - 1][column - 1] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 4:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row - 1][column - 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row + 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 5:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row][column - 1] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row - 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 6:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row][column - 1] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row + 1][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 7:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row - 1][column] == 0 and chessboard[row - 1][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '503':
            if orientation == 0:
                if row > 0 and row < ROW - 1 and column > 0 and column < COLUMN - 1:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row - 1][column] == 0 and chessboard[row][column + 1] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        elif chess == '507':
            if orientation == 0:
                if row > 0 and column > 0 and column < COLUMN - 2:
                    if chessboard[row][column - 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row][column + 2] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 1:
                if row > 0 and row < ROW - 2 and column < COLUMN - 1:
                    if chessboard[row - 1][column] == 0 and chessboard[row][column + 1] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row + 2][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 2:
                if row < ROW - 1 and column > 1 and column < COLUMN - 1:
                    if chessboard[row][column + 1] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row][column - 1] == 0 and chessboard[row][column - 2] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 3:
                if row > 1 and row < ROW - 1 and column > 0:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row - 1][column] == 0 and chessboard[row - 2][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 4:
                if row > 0 and column > 1 and column < COLUMN - 1:
                    if chessboard[row][column + 1] == 0 and chessboard[row - 1][column] == 0 and \
                        chessboard[row][column - 1] == 0 and chessboard[row][column - 2] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 5:
                if row > 1 and row < ROW - 1 and column < COLUMN - 1:
                    if chessboard[row + 1][column] == 0 and chessboard[row][column + 1] == 0 and \
                        chessboard[row - 1][column] == 0 and chessboard[row - 2][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 6:
                if row < ROW - 1 and column > 0 and column < COLUMN - 2:
                    if chessboard[row][column - 1] == 0 and chessboard[row + 1][column] == 0 and \
                        chessboard[row][column + 1] == 0 and chessboard[row][column + 2] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            elif orientation == 7:
                if row > 0 and row < ROW - 2 and column > 0:
                    if chessboard[row - 1][column] == 0 and chessboard[row][column - 1] == 0 and \
                        chessboard[row + 1][column] == 0 and chessboard[row + 2][column] == 0:
                        return 1
                    else:
                        return 0
                else:
                    return 0
            else:
                return -1
        else:
            return -1
    else:
        #the red circle block invalid
        return 0

def angle_check(chessboard, row, column, chess, orientation, playerID):
#This function will check whether the step is angle valid or not
#Please do overlap check befor angle check!
#Can do first step check
    #Board expand
    e_chessboard = [[0] * (COLUMN + 2) for i in range(ROW + 2)]
    e_chessboard[0][0] = PID1
    e_chessboard[0][COLUMN + 1] = PID2
    e_chessboard[ROW + 1][COLUMN + 1] = PID3
    e_chessboard[ROW + 1][0] = PID4
    for i in range(ROW):
        e_chessboard[i + 1][1:(COLUMN + 1)] = chessboard[i][:]
    e_row = row + 1;
    e_column = column + 1;
    ID = playerID
    
    #check angle first, check edge later
    if chess == '101':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 1] == ID or e_chessboard[e_row - 1][e_column + 1] == ID or \
                e_chessboard[e_row + 1][e_column - 1] == ID or e_chessboard[e_row + 1][e_column + 1] == ID:
                if e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row + 1][e_column] != ID:
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return -1
    elif chess == '201':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 1] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row + 1][e_column - 1] == ID or e_chessboard[e_row + 1][e_column + 2] == ID:
                if e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row][e_column + 2] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 1][e_column - 1] == ID or e_chessboard[e_row - 1][e_column + 1] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 2][e_column + 1] == ID:
                if e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '302':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row + 1][e_column - 2] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 2][e_column + 1] == ID or \
                e_chessboard[e_row - 1][e_column + 1] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row + 1][e_column - 2] == ID or \
                e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row + 1][e_column + 1] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row - 2][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row + 1][e_column - 1] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row - 2][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 2][e_column + 1] == ID or \
                e_chessboard[e_row - 1][e_column - 1] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0
            else:
                return 0
        else:
            return -1
    elif chess == '301':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row + 1][e_column + 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row][e_column + 2] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 2][e_column + 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '405':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 1] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 2][e_column + 2] == ID:
                if e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 2] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '402':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row - 2][e_column - 1] == ID or \
                e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row + 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 2][e_column + 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row + 2][e_column - 1] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row + 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row + 2][e_column - 1] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row - 1][e_column - 2] == ID or \
                e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 2][e_column + 1] == ID:
                if e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '401':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row - 1][e_column + 3] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row + 1][e_column + 3] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row][e_column + 3] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 2] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row + 3][e_column - 1] == ID or e_chessboard[e_row + 3][e_column + 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row + 3][e_column] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 2][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '404':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row - 2][e_column] == ID or \
                e_chessboard[e_row - 2][e_column + 2] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 2][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 2] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column + 2] == ID or e_chessboard[e_row + 2][e_column - 1] == ID or \
                e_chessboard[e_row - 2][e_column - 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 2] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row + 2][e_column] == ID or \
                e_chessboard[e_row + 2][e_column - 2] == ID or e_chessboard[e_row - 1][e_column - 2] == ID or \
                e_chessboard[e_row - 1][e_column + 2] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row + 2][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 2] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row][e_column - 2] == ID or \
                e_chessboard[e_row - 2][e_column - 2] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID:
                if e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 2] != ID and \
                    e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 2][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 4:
            if e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row - 2][e_column] == ID or \
                e_chessboard[e_row - 2][e_column - 2] == ID or e_chessboard[e_row + 1][e_column - 2] == ID or \
                e_chessboard[e_row + 1][e_column + 2] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 2][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 2] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 5:
            if e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row][e_column + 2] == ID or \
                e_chessboard[e_row - 2][e_column + 2] == ID or e_chessboard[e_row - 2][e_column - 1] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID:
                if e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 2] != ID and \
                    e_chessboard[e_row - 2][e_column + 1] != ID and e_chessboard[e_row - 2][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 6:
            if e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row + 2][e_column] == ID or \
                e_chessboard[e_row + 2][e_column + 2] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row - 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row + 2][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 2] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 7:
            if e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row][e_column - 2] == ID or \
                e_chessboard[e_row + 2][e_column - 2] == ID or e_chessboard[e_row + 2][e_column + 1] == ID or \
                e_chessboard[e_row - 2][e_column + 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 2] != ID and \
                    e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '403':
        if orientation == 0:
            if e_chessboard[e_row][e_column - 2] == ID or e_chessboard[e_row - 1][e_column - 1] == ID or \
                e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row + 2][e_column - 2] == ID:
                if e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 2][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 2][e_column] == ID or e_chessboard[e_row - 1][e_column + 1] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row + 2][e_column - 1] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row - 2][e_column - 2] == ID:
                if e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 2] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row][e_column + 2] == ID or e_chessboard[e_row - 1][e_column + 1] == ID or \
                e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row + 1][e_column - 2] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 2][e_column + 2] == ID:
                if e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 2][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row + 2][e_column] == ID or e_chessboard[e_row + 1][e_column + 1] == ID or \
                e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row - 2][e_column - 1] == ID or \
                e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row + 2][e_column - 2] == ID:
                if e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 2] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '508':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column + 4] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row + 1][e_column - 1] == ID or \
                e_chessboard[e_row + 1][e_column + 4] == ID:
                if e_chessboard[e_row][e_column + 4] != ID and e_chessboard[e_row - 1][e_column + 3] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 2] != ID and \
                    e_chessboard[e_row + 1][e_column + 3] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row + 4][e_column + 1] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row - 1][e_column - 1] == ID or \
                e_chessboard[e_row + 4][e_column - 1] == ID:
                if e_chessboard[e_row + 4][e_column] != ID and e_chessboard[e_row + 3][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row + 2][e_column - 1] != ID and \
                    e_chessboard[e_row + 3][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row + 1][e_column - 4] == ID or e_chessboard[e_row + 2][e_column - 1] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 1] == ID or \
                e_chessboard[e_row - 1][e_column - 4] == ID:
                if e_chessboard[e_row][e_column - 4] != ID and e_chessboard[e_row + 1][e_column - 3] != ID and \
                    e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 2] != ID and \
                    e_chessboard[e_row - 1][e_column - 3] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row - 4][e_column - 1] == ID or e_chessboard[e_row - 1][e_column - 2] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row + 1][e_column + 1] == ID or \
                e_chessboard[e_row - 4][e_column + 1] == ID:
                if e_chessboard[e_row - 4][e_column] != ID and e_chessboard[e_row - 3][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row - 2][e_column + 1] != ID and \
                    e_chessboard[e_row - 3][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 4:
            if e_chessboard[e_row - 1][e_column - 4] == ID or e_chessboard[e_row - 2][e_column - 1] == ID or \
                e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row + 1][e_column + 1] == ID or \
                e_chessboard[e_row + 1][e_column - 4] == ID:
                if e_chessboard[e_row][e_column - 4] != ID and e_chessboard[e_row - 1][e_column - 3] != ID and \
                    e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 2] != ID and \
                    e_chessboard[e_row + 1][e_column - 3] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 5:
            if e_chessboard[e_row - 4][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row + 1][e_column - 1] == ID or \
                e_chessboard[e_row - 4][e_column - 1] == ID:
                if e_chessboard[e_row - 4][e_column] != ID and e_chessboard[e_row - 3][e_column + 1] != ID and \
                    e_chessboard[e_row - 2][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row - 2][e_column - 1] != ID and \
                    e_chessboard[e_row - 3][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 6:
            if e_chessboard[e_row + 1][e_column + 4] == ID or e_chessboard[e_row + 2][e_column + 1] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row - 1][e_column - 1] == ID or \
                e_chessboard[e_row - 1][e_column + 4] == ID:
                if e_chessboard[e_row][e_column + 4] != ID and e_chessboard[e_row + 1][e_column + 3] != ID and \
                    e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 2] != ID and \
                    e_chessboard[e_row - 1][e_column + 3] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 7:
            if e_chessboard[e_row + 4][e_column - 1] == ID or e_chessboard[e_row + 1][e_column - 2] == ID or \
                e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row - 1][e_column + 1] == ID or \
                e_chessboard[e_row + 4][e_column + 1] == ID:
                if e_chessboard[e_row + 4][e_column] != ID and e_chessboard[e_row + 3][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row + 2][e_column + 1] != ID and \
                    e_chessboard[e_row + 3][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '502':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row - 3][e_column - 1] == ID or \
                e_chessboard[e_row - 3][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row + 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 3][e_column] != ID and \
                    e_chessboard[e_row - 2][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 3] == ID or \
                e_chessboard[e_row + 1][e_column + 3] == ID or e_chessboard[e_row + 2][e_column + 1] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row - 2][e_column - 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 3] != ID and \
                    e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row + 3][e_column - 1] == ID or \
                e_chessboard[e_row + 3][e_column + 1] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row - 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 3][e_column] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 1][e_column - 3] == ID or \
                e_chessboard[e_row + 1][e_column - 3] == ID or e_chessboard[e_row + 2][e_column - 1] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row - 2][e_column + 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 3] != ID and \
                    e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '504':
        if orientation == 0:
            if e_chessboard[e_row - 3][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 3] == ID or \
                e_chessboard[e_row + 1][e_column + 3] == ID or e_chessboard[e_row + 1][e_column - 1] == ID or \
                e_chessboard[e_row - 3][e_column - 1] == ID:
                if e_chessboard[e_row - 3][e_column] != ID and e_chessboard[e_row - 2][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 2] != ID and \
                    e_chessboard[e_row][e_column + 3] != ID and e_chessboard[e_row + 1][e_column + 2] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row + 1][e_column + 3] == ID or e_chessboard[e_row + 3][e_column + 1] == ID or \
                e_chessboard[e_row + 3][e_column - 1] == ID or e_chessboard[e_row - 1][e_column - 1] == ID or \
                e_chessboard[e_row - 1][e_column + 3] == ID:
                if e_chessboard[e_row][e_column + 3] != ID and e_chessboard[e_row + 1][e_column + 2] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row + 2][e_column + 1] != ID and \
                    e_chessboard[e_row + 3][e_column] != ID and e_chessboard[e_row + 2][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row + 1][e_column - 3] == ID or e_chessboard[e_row + 3][e_column - 1] == ID or \
                e_chessboard[e_row + 3][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 1] == ID or \
                e_chessboard[e_row - 1][e_column - 3] == ID:
                if e_chessboard[e_row][e_column - 3] != ID and e_chessboard[e_row + 1][e_column - 2] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row + 2][e_column - 1] != ID and \
                    e_chessboard[e_row + 3][e_column] != ID and e_chessboard[e_row + 2][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 2] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row - 3][e_column - 1] == ID or e_chessboard[e_row - 1][e_column - 3] == ID or \
                e_chessboard[e_row + 1][e_column - 3] == ID or e_chessboard[e_row + 1][e_column + 1] == ID or \
                e_chessboard[e_row - 3][e_column + 1] == ID:
                if e_chessboard[e_row - 3][e_column] != ID and e_chessboard[e_row - 2][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 2] != ID and \
                    e_chessboard[e_row][e_column - 3] != ID and e_chessboard[e_row + 1][e_column - 2] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 2][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '510':
        if orientation == 0:
            if e_chessboard[e_row][e_column - 2] == ID or e_chessboard[e_row - 1][e_column - 1] == ID or \
                e_chessboard[e_row - 1][e_column + 3] == ID or e_chessboard[e_row + 1][e_column + 3] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row + 2][e_column - 2] == ID:
                if e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 3] != ID and \
                    e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 2][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 2][e_column] == ID or e_chessboard[e_row - 1][e_column + 1] == ID or \
                e_chessboard[e_row + 3][e_column + 1] == ID or e_chessboard[e_row + 3][e_column - 1] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row - 2][e_column - 2] == ID:
                if e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 3][e_column] != ID and \
                    e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 2] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row][e_column + 2] == ID or e_chessboard[e_row + 1][e_column + 1] == ID or \
                e_chessboard[e_row + 1][e_column - 3] == ID or e_chessboard[e_row - 1][e_column - 3] == ID or \
                e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 2][e_column + 2] == ID:
                if e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 3] != ID and \
                    e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 2][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row + 2][e_column] == ID or e_chessboard[e_row + 1][e_column - 1] == ID or \
                e_chessboard[e_row - 3][e_column - 1] == ID or e_chessboard[e_row - 3][e_column + 1] == ID or \
                e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row + 2][e_column + 2] == ID:
                if e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 3][e_column] != ID and \
                    e_chessboard[e_row - 2][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 2] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 4:
            if e_chessboard[e_row][e_column + 2] == ID or e_chessboard[e_row - 1][e_column + 1] == ID or \
                e_chessboard[e_row - 1][e_column - 3] == ID or e_chessboard[e_row + 1][e_column - 3] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 2][e_column + 2] == ID:
                if e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 3] != ID and \
                    e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 2][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 5:
            if e_chessboard[e_row + 2][e_column] == ID or e_chessboard[e_row + 1][e_column + 1] == ID or \
                e_chessboard[e_row - 3][e_column + 1] == ID or e_chessboard[e_row - 3][e_column - 1] == ID or \
                e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row + 2][e_column - 2] == ID:
                if e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 2][e_column + 1] != ID and e_chessboard[e_row - 3][e_column] != ID and \
                    e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 2] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 6:
            if e_chessboard[e_row][e_column - 2] == ID or e_chessboard[e_row + 1][e_column - 1] == ID or \
                e_chessboard[e_row + 1][e_column + 3] == ID or e_chessboard[e_row - 1][e_column + 3] == ID or \
                e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row - 2][e_column - 2] == ID:
                if e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 3] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 2][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 7:
            if e_chessboard[e_row - 2][e_column] == ID or e_chessboard[e_row - 1][e_column - 1] == ID or \
                e_chessboard[e_row + 3][e_column - 1] == ID or e_chessboard[e_row + 3][e_column + 1] == ID or \
                e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row - 2][e_column + 2] == ID:
                if e_chessboard[e_row - 2][e_column + 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 3][e_column] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 2] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '511':
        if orientation == 0:
            if e_chessboard[e_row + 2][e_column - 2] == ID or e_chessboard[e_row - 1][e_column - 2] == ID or \
                e_chessboard[e_row - 2][e_column] == ID or e_chessboard[e_row - 2][e_column + 2] == ID or \
                e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row + 2][e_column] == ID:
                if e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 2] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 2][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 2] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row + 1][e_column] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 2][e_column - 2] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row][e_column + 2] == ID or e_chessboard[e_row + 2][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row][e_column - 2] == ID:
                if e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row - 2][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 2] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row + 2][e_column + 2] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row - 2][e_column] == ID or e_chessboard[e_row - 2][e_column - 2] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row + 2][e_column] == ID:
                if e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 2] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 2][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 2] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row + 1][e_column] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row - 2][e_column + 2] == ID or e_chessboard[e_row - 2][e_column - 1] == ID or \
                e_chessboard[e_row][e_column - 2] == ID or e_chessboard[e_row + 2][e_column - 2] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row][e_column + 2] == ID:
                if e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row - 2][e_column + 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 2] != ID and \
                    e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '501':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 3] == ID or e_chessboard[e_row - 1][e_column + 3] == ID or \
                e_chessboard[e_row + 1][e_column + 3] == ID or e_chessboard[e_row + 1][e_column - 3] == ID:
                if e_chessboard[e_row][e_column - 3] != ID and e_chessboard[e_row - 1][e_column - 2] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 2] != ID and \
                    e_chessboard[e_row][e_column + 3] != ID and e_chessboard[e_row + 1][e_column + 2] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 2] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 3][e_column + 1] == ID or e_chessboard[e_row + 3][e_column + 1] == ID or \
                e_chessboard[e_row + 3][e_column - 1] == ID or e_chessboard[e_row - 3][e_column - 1] == ID:
                if e_chessboard[e_row - 3][e_column] != ID and e_chessboard[e_row - 2][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row + 2][e_column + 1] != ID and \
                    e_chessboard[e_row + 3][e_column] != ID and e_chessboard[e_row + 2][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row - 2][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '506':
        if orientation == 0:
            if e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column + 2] == ID or e_chessboard[e_row + 2][e_column - 1] == ID or \
                e_chessboard[e_row - 2][e_column - 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 2] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row + 2][e_column + 1] == ID or \
                e_chessboard[e_row + 2][e_column - 2] == ID or e_chessboard[e_row - 1][e_column - 2] == ID or \
                e_chessboard[e_row - 1][e_column + 2] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 2][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 2] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 1][e_column - 2] == ID or \
                e_chessboard[e_row - 2][e_column - 2] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID:
                if e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 2] != ID and \
                    e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 2][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row - 2][e_column - 1] == ID or \
                e_chessboard[e_row - 2][e_column + 2] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 2][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 2] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 4:
            if e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 1][e_column - 2] == ID or \
                e_chessboard[e_row + 2][e_column - 2] == ID or e_chessboard[e_row + 2][e_column + 1] == ID or \
                e_chessboard[e_row - 2][e_column + 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 2] != ID and \
                    e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 5:
            if e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row - 2][e_column - 2] == ID or e_chessboard[e_row + 1][e_column - 2] == ID or \
                e_chessboard[e_row + 1][e_column + 2] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 2][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 2] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 6:
            if e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row - 2][e_column + 2] == ID or e_chessboard[e_row - 2][e_column - 1] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID:
                if e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 2] != ID and \
                    e_chessboard[e_row - 2][e_column + 1] != ID and e_chessboard[e_row - 2][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 7:
            if e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row + 2][e_column - 1] == ID or \
                e_chessboard[e_row + 2][e_column + 2] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row - 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 2][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 2] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '505':
        if orientation == 0:
            if e_chessboard[e_row + 2][e_column - 2] == ID or e_chessboard[e_row - 1][e_column - 2] == ID or \
                e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 2][e_column + 2] == ID or \
                e_chessboard[e_row][e_column + 2] == ID or e_chessboard[e_row + 1][e_column + 1] == ID or \
                e_chessboard[e_row + 2][e_column] == ID:
                if e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 2] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 2][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 2][e_column - 2] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row + 2][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column] == ID or e_chessboard[e_row + 1][e_column - 1] == ID or \
                e_chessboard[e_row][e_column - 2] == ID:
                if e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row - 2][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 2] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row - 2][e_column + 2] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row + 2][e_column - 2] == ID or \
                e_chessboard[e_row][e_column - 2] == ID or e_chessboard[e_row - 1][e_column - 1] == ID or \
                e_chessboard[e_row - 2][e_column] == ID:
                if e_chessboard[e_row - 2][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 2] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 2][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row + 2][e_column + 2] == ID or e_chessboard[e_row + 2][e_column - 1] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row - 2][e_column - 2] == ID or \
                e_chessboard[e_row - 2][e_column] == ID or e_chessboard[e_row - 1][e_column + 1] == ID or \
                e_chessboard[e_row][e_column + 2] == ID:
                if e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row + 2][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 2] != ID and \
                    e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '509':
        if orientation == 0:
            if e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 2][e_column + 2] == ID or \
                e_chessboard[e_row][e_column + 2] == ID or e_chessboard[e_row + 2][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 2][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row + 2][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row + 2][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column] == ID or e_chessboard[e_row + 2][e_column - 2] == ID or \
                e_chessboard[e_row - 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 2] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 2] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row - 2][e_column - 2] == ID or \
                e_chessboard[e_row][e_column - 2] == ID or e_chessboard[e_row + 2][e_column - 2] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 2][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row + 2][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row - 2][e_column + 2] == ID or \
                e_chessboard[e_row - 2][e_column] == ID or e_chessboard[e_row - 2][e_column - 2] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 2] != ID and \
                    e_chessboard[e_row - 2][e_column + 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 2] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '512':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row - 2][e_column - 1] == ID or \
                e_chessboard[e_row - 2][e_column + 2] == ID or e_chessboard[e_row][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row + 2][e_column - 1] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 2][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column + 2] == ID or e_chessboard[e_row + 2][e_column] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row - 1][e_column - 2] == ID or \
                e_chessboard[e_row - 2][e_column - 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 2] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row][e_column - 2] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row + 2][e_column + 1] == ID or \
                e_chessboard[e_row + 2][e_column - 2] == ID or e_chessboard[e_row][e_column - 2] == ID or \
                e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row - 1][e_column + 2] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 2][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row - 2][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 1][e_column - 2] == ID or \
                e_chessboard[e_row - 2][e_column - 2] == ID or e_chessboard[e_row - 2][e_column] == ID or \
                e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID:
                if e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 2] != ID and \
                    e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row][e_column + 2] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 4:
            if e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row - 2][e_column - 2] == ID or e_chessboard[e_row][e_column - 2] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 2][e_column + 1] == ID or \
                e_chessboard[e_row + 1][e_column + 2] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 2][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID and e_chessboard[e_row + 2][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 5:
            if e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row - 2][e_column + 2] == ID or e_chessboard[e_row - 2][e_column] == ID or \
                e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row + 1][e_column - 2] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID:
                if e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 2] != ID and \
                    e_chessboard[e_row - 2][e_column + 1] != ID and e_chessboard[e_row - 1][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID and e_chessboard[e_row][e_column - 2] != ID and \
                    e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 6:
            if e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row + 2][e_column - 1] == ID or \
                e_chessboard[e_row + 2][e_column + 2] == ID or e_chessboard[e_row][e_column + 2] == ID or \
                e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row - 2][e_column - 1] == ID or \
                e_chessboard[e_row - 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 2][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID and e_chessboard[e_row - 2][e_column] != ID and \
                    e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 7:
            if e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 1][e_column - 2] == ID or \
                e_chessboard[e_row + 2][e_column - 2] == ID or e_chessboard[e_row + 2][e_column] == ID or \
                e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row - 2][e_column + 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 2] != ID and \
                    e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 1][e_column] != ID and \
                    e_chessboard[e_row + 1][e_column + 1] != ID and e_chessboard[e_row][e_column + 2] != ID and \
                    e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '503':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row - 2][e_column - 1] == ID or \
                e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row + 2][e_column + 1] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    elif chess == '507':
        if orientation == 0:
            if e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row - 2][e_column - 1] == ID or \
                e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 3] == ID or \
                e_chessboard[e_row + 1][e_column + 3] == ID or e_chessboard[e_row + 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 3] != ID and \
                    e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 1:
            if e_chessboard[e_row - 2][e_column + 1] == ID or e_chessboard[e_row - 1][e_column + 2] == ID or \
                e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row + 3][e_column + 1] == ID or \
                e_chessboard[e_row + 3][e_column - 1] == ID or e_chessboard[e_row - 2][e_column - 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 3][e_column] != ID and \
                    e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 2:
            if e_chessboard[e_row + 1][e_column + 2] == ID or e_chessboard[e_row + 2][e_column + 1] == ID or \
                e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 1][e_column - 3] == ID or \
                e_chessboard[e_row - 1][e_column - 3] == ID or e_chessboard[e_row - 1][e_column + 2] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 3] != ID and \
                    e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 3:
            if e_chessboard[e_row + 2][e_column - 1] == ID or e_chessboard[e_row + 1][e_column - 2] == ID or \
                e_chessboard[e_row - 1][e_column - 2] == ID or e_chessboard[e_row - 3][e_column - 1] == ID or \
                e_chessboard[e_row - 3][e_column + 1] == ID or e_chessboard[e_row + 2][e_column + 1] == ID:
                if e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 3][e_column] != ID and \
                    e_chessboard[e_row - 2][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 4:
            if e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row - 2][e_column + 1] == ID or \
                e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 1][e_column - 3] == ID or \
                e_chessboard[e_row + 1][e_column - 3] == ID or e_chessboard[e_row + 1][e_column + 2] == ID:
                if e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row - 1][e_column - 2] != ID and e_chessboard[e_row][e_column - 3] != ID and \
                    e_chessboard[e_row + 1][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 1][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 5:
            if e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row + 1][e_column + 2] == ID or \
                e_chessboard[e_row - 1][e_column + 2] == ID or e_chessboard[e_row - 3][e_column + 1] == ID or \
                e_chessboard[e_row - 3][e_column - 1] == ID or e_chessboard[e_row + 2][e_column - 1] == ID:
                if e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 2][e_column + 1] != ID and e_chessboard[e_row - 3][e_column] != ID and \
                    e_chessboard[e_row - 2][e_column - 1] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 1] != ID and e_chessboard[e_row + 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 6:
            if e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row + 2][e_column - 1] == ID or \
                e_chessboard[e_row + 2][e_column + 1] == ID or e_chessboard[e_row + 1][e_column + 3] == ID or \
                e_chessboard[e_row - 1][e_column + 3] == ID or e_chessboard[e_row - 1][e_column - 2] == ID:
                if e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row + 1][e_column + 2] != ID and e_chessboard[e_row][e_column + 3] != ID and \
                    e_chessboard[e_row - 1][e_column + 2] != ID and e_chessboard[e_row - 1][e_column + 1] != ID and \
                    e_chessboard[e_row - 1][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        elif orientation == 7:
            if e_chessboard[e_row - 2][e_column - 1] == ID or e_chessboard[e_row - 1][e_column - 2] == ID or \
                e_chessboard[e_row + 1][e_column - 2] == ID or e_chessboard[e_row + 3][e_column - 1] == ID or \
                e_chessboard[e_row + 3][e_column + 1] == ID or e_chessboard[e_row - 2][e_column + 1] == ID:
                if e_chessboard[e_row - 2][e_column] != ID and e_chessboard[e_row - 1][e_column - 1] != ID and \
                    e_chessboard[e_row][e_column - 2] != ID and e_chessboard[e_row + 1][e_column - 1] != ID and \
                    e_chessboard[e_row + 2][e_column - 1] != ID and e_chessboard[e_row + 3][e_column] != ID and \
                    e_chessboard[e_row + 2][e_column + 1] != ID and e_chessboard[e_row + 1][e_column + 1] != ID and \
                    e_chessboard[e_row][e_column + 1] != ID and e_chessboard[e_row - 1][e_column + 1] != ID:
                    return 1
                else:
                    return 0           
            else:
                return 0
        else:
            return -1
    else:
        return -1

def validity_check(chessboard, row, column, chess, orientation, playerID):
#valid=1, invalid=0
#this function will check the input step is valid or not for current chessboard
    if row < 0 or row >= ROW or column < 0 or column >= COLUMN:
        #invalid input
        return -1
    
    if 0 == overlap_check(chessboard, row, column, chess, orientation):
        return 0
    elif 0 == angle_check(chessboard, row, column, chess, orientation, playerID):
        return 0
    else:
        return 1
    
def step_validity_check(chessboard, playerID):
#valid=1, invalid=0
#this fuction will return an array to show all valid steps for current chessboard and plarer
#the output array will be 36400. If one of its elements equals to 1, it represet a valid step
    validity_steps = [0] * status_count
    #check corner first
    for row in range(ROW):
        for column in range(COLUMN):
            for chess in chessArray:
                for orientation in range(chessArray[chess]):
                    tmp = validity_check(chessboard, row, column, chess, orientation, playerID)
                    validity_steps[offsetArray[chess] + (row * COLUMN + column) * chessArray[chess] + orientation] = tmp
                    if tmp == 1:
                        print('Row: %d, Column: %d, Chess: %s, Orientation: %d' % (row, column, chess, orientation))
    return validity_steps


def chess2Array(chessboard, row, column, chess, orientation, playerID):
#input row/column/chess/orientation/playerID, output coresponding 36400 array 
    validity_steps = [0] * status_count
    
    if 1 == validity_check(chessboard, row, column, chess, orientation, playerID):
        validity_steps[offsetArray[chess] + (row * COLUMN + column) * chessArray[chess] + orientation] = 1;
        return validity_steps
    else:
        return -1
    
def playerRotate(chessboard, playerID):
#rotate current chessboard for differernt player
    tempboard = [[0] * COLUMN for x in range(ROW)]
    if playerID == PID1:
        pass
    elif playerID == PID2:
        for i in range(ROW):
            for j in range(COLUMN):
                tempboard[j][COLUMN - 1 - i] = chessboard[i][j]
        for i in range(ROW):
            for j in range(COLUMN):
                chessboard[i][j] = tempboard[i][j]
    elif playerID == PID3:
        for i in range(ROW):
            for j in range(COLUMN):
                tempboard[ROW - 1 - i][COLUMN - 1 - j] = chessboard[i][j]
        for i in range(ROW):
            for j in range(COLUMN):
                chessboard[i][j] = tempboard[i][j]
    elif playerID == PID4:
        for i in range(ROW):
            for j in range(COLUMN):
                tempboard[ROW - 1 - j][i] = chessboard[i][j]
        for i in range(ROW):
            for j in range(COLUMN):
                chessboard[i][j] = tempboard[i][j]
  