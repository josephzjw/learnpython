# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:17:39 2017

@author: z81022682
test myClient
"""

# client  
  
import socket  
import sys
import json
#import time

def myClient(myID,serverIP,serverPort):
    
    ##### connect the server #####
    address = (serverIP,serverPort)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(address)
    print("Connect Server Success!")
    
    ##### send information to sever #####
    msg = '{"msg_name":"registration","msg_data":{"team_id":%d,"team_name":"TTT"}}'%(myID)
    msg_send = "%05d%s"%(len(msg),msg)
    s.send(msg_send.encode('ascii'))
    print('Register my info to server success')
    
    ##### start the game #####
    while 1:
        dataIn = s.recv(512)
    
        dataJson = json.loads(dataIn[5:])
        print(dataJson)
        if dataJson["msg_name"] == "inquire":
#            print('server is asking for data!')
            
            msg = '{"msg_name":"action","msg_data":{"hand_no":1,"team_id":%d,"player_id":1,"chessman":{"id":301,"squareness":[{"x":0,"y":0},{"x":0,"y":1},{"x":0,"y":2}]}}}'%(myID)
            msg_send = "%05d%s"%(len(msg),msg)
            s.send(msg_send.encode('ascii'))
#            print(dataJson["msg_data"]["players"][0]["player_id"])
        if dataJson['msg_name'] == 'inquire' and dataJson['msg_data']['player_id'] == 2:
            msg = '{"msg_name":"action","msg_data":{"hand_no":2,"team_id":%d,"player_id":2,"chessman":{"id":301,"squareness":[{"x":0,"y":19},{"x":0,"y":18},{"x":0,"y":17}]}}}'%(myID)
            msg_send = "%05d%s"%(len(msg),msg)
            s.send(msg_send.encode('ascii'))
    s.close()
    
#    s.send(('hihi').encode('utf-8'))
    
    s.close()
    
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python %s Team_ID IP Port'%(sys.argv[0]) )
        exit(1)
        
    f_ip = sys.argv[2]
    f_port = sys.argv[3]
    myID = int(sys.argv[1])
    
    myClient(myID,f_ip,int(f_port))

#address = ('127.0.0.1', 31500)  
#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
#s.connect(address)  
#  
#data = s.recv(512).decode('utf-8')  
#print('the data received is',data)  
#  
#s.send(('hihi').encode('utf-8'))  
#  
#s.close() 
