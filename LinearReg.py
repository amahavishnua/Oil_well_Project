#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 00:24:51 2020

@author: mvr
"""
#depth,coal,temp,sand
import io
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import numpy as np 
from sklearn import datasets, linear_model, metrics 
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
with open("/Users/mvr/Desktop/Well_logs/05.PETROPHYSICAL INTERPRETATION/15_9-19 BT2/CPI/15_9-19_BT2_CPI.las") as file:
    openedfile = file.read()
    
    
#print(type(openedfile))
#columns = [col1,col2,col3]
#print(ooopenedfile)
#data = io.StringIO(openedfile)
#dataa = pd.read_csv(data, skiprows=2,sep=" ")
c=0
print(type(openedfile))
print(openedfile)
for line in openedfile.split('\n'):
    if c <50:
        print(line)
        c=c+1
count = 0
data = []


for line in openedfile.split('\n'):
    
    print(len(line.split()))
    sub_data = []
    #print(line)
    
    len_token = len(line.split())
    
    if(len_token != 14):
        continue
        #sub_data = ','.join(line.split())
        #sub_data += ','*(14-len_token)
    else:
        sub_data = ','.join(line.split())
    
    data.append(sub_data)

print(len(data)-3209)

#df = pd.DataFrame(data,columns=columns)


#dataa.head()
#df=pd.read_csv(data,sep=" ",error_bad_lines=False)
#df.shape
#openedfile=openedfile[:]
#df=pd.DataFrame(openedfile,columns=[['A','Depth','BVW','CARB_file','Coal_flag','KlogH','KlogV','phif','rohfl',
                                     #'roHma','RW','sand_flag','sw','temp','VSH']])
#seeing first rows
#openedfile.head()    
#print(openedfile.dtypes())
'''
conv_Array=np.asarray(openedfile)
X, Y = conv_Array[:], conv_Array[13].reshape(-1,1)
# Creatng a regressor 
reg_model = LinearRegression()
# putting data into model   
mdl = reg_model.fit(X, Y)
# extct the R^2
scores = reg_model.score(X,Y)

model = sm.OLS(Y, X)
results = model.fit()

'''