# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 20:04:23 2017

@author: z81022682
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import random
import myfunc as mf
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

myChessBoard = np.zeros((ROW,COLUMN,5),dtype=np.int)

#with open('20170828_141406.txt','r') as f:
#    execfile(f)

with open('20170828_141406.txt', 'r') as f:
     mydata = f.read()
     mydata_new = mydata[10:-1]
#    data = json.load(f)

data = json.loads(mydata_new)

numOfHand = len(data)-2 # the total steps of the chess game

myCurrentChessBoard = np.zeros((20,20,5,numOfHand),dtype=np.int) # where to put the current squareness

##### myCurrentChessState: qi pan zhuang tai
myCurrentChessState = np.zeros((20,20,numOfHand+1),dtype=np.int) # what does the current chessboard look like
myFriendEnemyState = np.zeros((20,20,4,numOfHand),dtype=np.int) # 0:null,1:me,

######## every step output
myAxisLabel = []
#my_data = json.load(open("20170828_141406.txt").read())

numOfCause = 0

for i in range(numOfHand): # the i-th hand
    if i != 0:
        myCurrentChessState[:,:,i+1] = myCurrentChessState[:,:,i]
        
    if data[i+1]['msg_data']['player_id'] == 1:
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==1) # wofang
        myFriendEnemyState[ix,iy,1,i] = 1
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==2) # difang
        myFriendEnemyState[ix,iy,3,i] = 1
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==4) # difang
        myFriendEnemyState[ix,iy,3,i] = 1
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==3) # youfang
        myFriendEnemyState[ix,iy,2,i] = 1
    elif data[i+1]['msg_data']['player_id'] == 2:
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==2) # wofang
        myFriendEnemyState[ix,iy,1,i] = 1
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==3) # difang
        myFriendEnemyState[ix,iy,3,i] = 1
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==1) # difang
        myFriendEnemyState[ix,iy,3,i] = 1
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==4) # youfang
        myFriendEnemyState[ix,iy,2,i] = 1
    elif data[i+1]['msg_data']['player_id'] == 3:
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==3) # wofang
        myFriendEnemyState[ix,iy,1,i] = 1
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==4) # difang
        myFriendEnemyState[ix,iy,3,i] = 1
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==2) # difang
        myFriendEnemyState[ix,iy,3,i] = 1
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==1) # youfang
        myFriendEnemyState[ix,iy,2,i] = 1
    elif data[i+1]['msg_data']['player_id'] == 4:
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==4) # wofang
        myFriendEnemyState[ix,iy,1,i] = 1
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==1) # difang
        myFriendEnemyState[ix,iy,3,i] = 1
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==3) # difang
        myFriendEnemyState[ix,iy,3,i] = 1
        [ix,iy] = np.where(myCurrentChessState[:,:,i]==2) # youfang
        myFriendEnemyState[ix,iy,2,i] = 1
        
                
    if bool(data[i+1]['msg_data']['chessman'].get('squareness')): # this player can play
        numOfSquareness = len(data[i+1]['msg_data']['chessman']['squareness'])
        myTmpAxis = []
        if numOfSquareness >1:
            for j in range(numOfSquareness):
                xid = data[i+1]['msg_data']['chessman']['squareness'][j]['x'] # x_axis
                yid = data[i+1]['msg_data']['chessman']['squareness'][j]['y'] # y_axis
                zid = data[i+1]['msg_data']['player_id'] # player_id
                myCurrentChessBoard[xid,yid,zid,i]=1
                myCurrentChessState[xid,yid,i+1] = zid
                myTmpAxis.append([xid,yid])
        else:
            myTmpAxis = [[data[i+1]['msg_data']['chessman']['squareness'][0]['x'],data[i+1]['msg_data']['chessman']['squareness'][0]['y']]]
        myTmpAxisSorted = sorted(myTmpAxis)
        myAxisLabel.append(mf.axis2label(myTmpAxisSorted))
    else:
        myAxisLabel.append(-1)

"""
myPlot(myCurrentChessState[:, :, 4])

chessboard = myCurrentChessState[:, :, 4].tolist()

validity_steps = step_validity_check(chessboard, PID1)

next_step = chess2Array(0, 3, '101', 0, PID1)

myplt(label2axis(next_step.index(1)))
"""
tempboard = [[0] * (COLUMN) for k in range(ROW)]
chessboard = [[0] * (ROW * COLUMN) for k in range(50)]
chessboard3D = [[0] * (ROW * COLUMN * 3) for k in range(50)]
currentChessState3D = [[0] * (ROW * COLUMN * 3) for k in range(numOfHand + 1)]
onehotcurrentChessState = [[0] * (ROW * COLUMN) for k in range(numOfHand + 1)]

ID = PID1
'''
for i in range(numOfHand + 1):
    onehotcurrentChessState[i][:] = sum(myCurrentChessState[:, :, i].tolist(), [])
    for j in range(ROW * COLUMN):
        onehotcurrentChessState[i][j] = 1 << onehotcurrentChessState[i][j]
'''

for i in range(numOfHand + 1):
    for k1 in range(ROW):
        for k2 in range(COLUMN):
            if myCurrentChessState[k1, k2, i] == 0:
                currentChessState3D[i][(k1 * COLUMN + k2) * 3 + 0] = 0
                currentChessState3D[i][(k1 * COLUMN + k2) * 3 + 1] = 0
                currentChessState3D[i][(k1 * COLUMN + k2) * 3 + 2] = 0
            elif myCurrentChessState[k1, k2, i] == 1:
                currentChessState3D[i][(k1 * COLUMN + k2) * 3 + 0] = 1
                currentChessState3D[i][(k1 * COLUMN + k2) * 3 + 1] = 0
                currentChessState3D[i][(k1 * COLUMN + k2) * 3 + 2] = 0
            elif myCurrentChessState[k1, k2, i] == 3:
                currentChessState3D[i][(k1 * COLUMN + k2) * 3 + 0] = 0
                currentChessState3D[i][(k1 * COLUMN + k2) * 3 + 1] = 1
                currentChessState3D[i][(k1 * COLUMN + k2) * 3 + 2] = 0
            elif myCurrentChessState[k1, k2, i] == 2 or myCurrentChessState[k1, k2, i] == 4:
                currentChessState3D[i][(k1 * COLUMN + k2) * 3 + 0] = 0
                currentChessState3D[i][(k1 * COLUMN + k2) * 3 + 1] = 0
                currentChessState3D[i][(k1 * COLUMN + k2) * 3 + 2] = 1

sess = tf.InteractiveSession()
# weight initialization

#x = tf.placeholder("float", [None, ROW * COLUMN])
x = tf.placeholder("float", [None, ROW * COLUMN * 3])
y_ = tf.placeholder("float", shape=[None, status_count + 1])

W = tf.Variable(tf.zeros([ROW * COLUMN * 3,status_count + 1]))
b = tf.Variable(tf.zeros([status_count + 1]))
y = tf.nn.softmax(tf.matmul(x,W) + b)  

#x_image = tf.reshape(x, [-1,ROW,COLUMN,1])
x_image = tf.reshape(x, [-1,ROW,COLUMN,3])

#input to first hidden layer
#W_conv1 = weight_variable([5, 5, 1, 32])
W_conv1 = mf.weight_variable([5, 5, 3, 32])
b_conv1 = mf.bias_variable([32])
h_conv1 = tf.nn.relu(mf.conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = mf.max_pool_2x2(h_conv1)
#h_pool1 = h_conv1

#first hidden layer to second hidden layer
W_conv2 = mf.weight_variable([5, 5, 32, 64])
b_conv2 = mf.bias_variable([64])
h_conv2 = tf.nn.relu(mf.conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = mf.max_pool_2x2(h_conv2)

#second hidden layer to third hidden layer
W_fc1 = mf.weight_variable([(int(ROW/4) * int(COLUMN/4) * 64), 1024])
b_fc1 = mf.bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, int(ROW/4)*int(COLUMN/4)*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#output layer
W_fc2 = mf.weight_variable([1024, status_count + 1])
b_fc2 = mf.bias_variable([status_count + 1])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())



for i in range(5000):
    next_step = [[0] * (status_count + 1) for k in range(50)]
    for j in range(50):
        index = random.randint(0, numOfHand - 1)
        index = index // 4 * 4
        #randID = index % 4 + 1 
        if myAxisLabel[index] > 0:
            next_step[j][myAxisLabel[index]] = 1
        else:
            next_step[j][status_count] = 1
        #tempboard = myCurrentChessState[:,:,index].tolist()
        #playerRotate(tempboard, randID)
        #chessboard[j][:] = sum(tempboard, [])
        #chessboard[j][:] = onehotcurrentChessState[index][:]
        chessboard3D[j][:] = currentChessState3D[index][:]
                
      
    if i%25 == 0:
        #train_accuracy = accuracy.eval(feed_dict={
        #        x: chessboard, y_: next_step, keep_prob: 1.0})
        train_accuracy = accuracy.eval(feed_dict={
                x: chessboard3D, y_: next_step, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    #train_step.run(feed_dict={x: chessboard, y_: next_step, keep_prob: 0.5})
    train_step.run(feed_dict={x: chessboard3D, y_: next_step, keep_prob: 0.5})