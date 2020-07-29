#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:31:27 2020

@author: mvr
"""

import pandas as pd
import numpy as np

with open("/Users/mvr/Desktop/Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-19 BT2/CPI/15_9-19_BT2_CPI.las") as file:
    openedfile = file.read()
#print(type(openedfile))

'''
METHOD TO CHECK THE COLUMN NAMES OF THE OPENED FILE:

'''
c=0
for line in openedfile.split("\n"):
    if c <48:
        #print(line)
        c=c+1

coloumns=['Depth','BVW','CARB_FLAG','KLOGH','KLOGV','PHIF','ROFL','RHOMA','RW','SAND_FLAG','SW','TEMP','VSH']

pure_data=openedfile.split("\n")[47:]

'''
Snapshot of First 20 lines of cleaned_data


'''
for lin in pure_data[1:3]:
    #print("--"*70)
    print('')
    

trans_data=[]

for lin in range(len(pure_data)):
    #print("--"*70)
    #print(lin)
    
    temp_data = (pure_data[lin].split())
    trans_data.append(temp_data)
#print(trans_data[1])

count=0 

for i in trans_data:
    count=count+1
#print(count)

coloumns=['Depth','BVW','CARB_FLAG','Coal_flag','KLOGH','KLOGV','PHIF','ROFL','RHOMA','RW','SAND_FLAG','TEMP','VSH']

data_df=pd.DataFrame(trans_data,columns=['Depth','BVW','CARB_FLAG','Coal_flag','KLOGH','KLOGV','PHIF','ROFL','RHOMA','RW','SAND_FLAG','SW','TEMP','VSH'])


data_df.dropna(subset = ['Depth','BVW','CARB_FLAG','Coal_flag','KLOGH','KLOGV','PHIF','ROFL','RHOMA','RW','SAND_FLAG','SW','TEMP','VSH']
, inplace=True)

print(data_df.head())

################ MAKING PROCESS READY ##################


''' Creating an array with features and a seperate array for dependent SW'''
array = data_df.values
X1 = array[:,0:11]
X2=array[:,12:15]
Xa=np.concatenate((X1,X2),axis=1)
Ya = array[:,11:12]
Xa=Xa.astype(np.float) 
Ya=Ya.astype(np.float) 
print(Xa)
print(Ya)

# NOW CREATING DATA FRAME features

y=data_df['SW']
x=data_df.drop('SW',1)

################ DATA CLEANED AND CONVERTED ###########












