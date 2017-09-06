# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:16:08 2017

@author: z81022682

test python server
"""

# server  
  
import socket  

    


print('Waiting for the client!')
address = ('127.0.0.1', 31500)  
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # s = socket.socket()  
s.bind(address)  
s.listen(5)  
  
ss, addr = s.accept()  
print('got connected from',addr)  
  
ss.send(('byebye').encode('utf-8'))  
ra = ss.recv(512)  
print(ra.decode('utf-8'))  
  
ss.close()  
s.close() 