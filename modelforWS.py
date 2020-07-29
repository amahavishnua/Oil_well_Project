#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:19:26 2020

@author: mvr
"""

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
print(type(openedfile))

'''
METHOD TO CHECK THE COLUMN NAMES OF THE OPENED FILE:

'''
c=0
for line in openedfile.split("\n"):
    if c <48:
        print(line)
        c=c+1

coloumns=['Depth','BVW','CARB_FLAG','KLOGH','KLOGV','PHIF','ROFL','RHOMA','RW','SAND_FLAG','SW','TEMP','VSH']

pure_data=openedfile.split("\n")[47:]

'''
Snapshot of First 20 lines of cleaned_data


'''
for lin in pure_data[1:3]:
    print("--"*70)
    print(lin)
    

trans_data=[]
temp_data=[]
for lin in range(len(pure_data)):
    print("--"*70)
    print(lin)
    
    temp_data = (pure_data[lin].split())
    trans_data.append(temp_data)
print(trans_data[1])

count=0 

for i in trans_data:
    count=count+1
print(count)

coloumns=['Depth','BVW','CARB_FLAG','Coal_flag','KLOGH','KLOGV','PHIF','ROFL','RHOMA','RW','SAND_FLAG','TEMP','VSH']

df=pd.DataFrame(trans_data,columns=['Depth','BVW','CARB_FLAG','Coal_flag','KLOGH','KLOGV','PHIF','ROFL','RHOMA','RW','SAND_FLAG','SW','TEMP','VSH'])

df.head()
df.dropna(subset = ['Depth','BVW','CARB_FLAG','Coal_flag','KLOGH','KLOGV','PHIF','ROFL','RHOMA','RW','SAND_FLAG','SW','TEMP','VSH']
, inplace=True)

# Checking for any remaining NAN values
pd.isnull(df).sum() > 0
#pd.istr(df).sum() > 0
#df[df < 0] = 0

#Linear Regression Model
'''
['Depth','BVW','CARB_FLAG','Coal_flag','KLOGH',
'KLOGV','PHIF','ROFL','RHOMA','RW',
'SAND_FLAG','SW','TEMP','VSH'])

DEPTH.M                    :  
BVW  .V/V                  :  PHIF*SW
CARB_FLAG.UNITLESS                 :  Values changed using TEXT_EDIT
COAL_FLAG.UNITLESS                 :  Coal flag
KLOGH.MD                   :  Values changed using TEXT_EDIT
KLOGV.MD                   :  Vertical PERMEABILITY calculated from logs
PHIF .V/V                  :  Final porosity
RHOFL.G/CM3                 :  0.9
RHOMA.G/CM3                 :  Fixed RHOMA
RW   .OHMM                 :  Formation Resistivity
SAND_FLAG.UNITLESS                 :  Values changed using TEXT_EDIT
SW   .V/V                  :  Water Saturation
TEMP .DEGC                 :  Calculated temperature curve
VSH  .V/V                  :  (GR-8)/(100-8)

'''
#Target is SW
#Predictor variables : TEMP, PHIF, RHOMA, KLOGV, VSH, BVM

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression,LinearRegression
import sklearn
array = df.values
X1 = array[:,0:11]
X2=array[:,12:15]
X=np.concatenate((X1,X2),axis=1)
Y = array[:,11:12]


model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num_Features: %s" % (fit.n_features_))
print("selected_Features: %s" % (fit.support_))
print("Oilwell_Feature_Ranking: %s" % (fit.ranking_))

'''

Used logistic regression on the features, and the ranking is following

																Feature Ranking
SW   .V/V                  :  Water Saturation									our dependent variable


PHIF .V/V                          :  Final porosity						1
RHOMA.G/CM3                :  Fixed RHOMA								1
RW   .OHMM                     :  Formation Resistivity					1
BVW  .V/V                         :  PHIF*SW								2
SAND_FLAG.UNITLESS  :  Values changed using TEXT_EDIT				    3
VSH  .V/V 														        4
KLOGH.MD                       :  Values changed using TEXT_EDIT			5
CARB_FLAG.UNITLESS  :  Values changed using TEXT_EDIT				    6
TEMP .DEGC                 	  :  Calculated temperature curve			7
COAL_FLAG.UNITLESS  :  Coal flag									        8
KLOGV.MD                   	  :  Vertical PERMEABILITY calculated from logs		9
DEPTH.M                    :  												    10
RHOFL.G/CM3                 :  0.9										        11
 

Feature Ranking: [10  2  6  8  5  9  1 11  1  1  3  7  4]
'''

from sklearn.preprocessing import PolynomialFeatures

# Taking the three features PHIF,RHOMA,RW

F1= array[:,6:7]
F2= array[:,8:10]

Ranked_F=np.concatenate((F2,F1),axis=1)

Y = array[:,11:12]


# Tinkering the feature data to fit into the model (Tranform)
RF_T = PolynomialFeatures(degree=3, include_bias=False).fit_transform(Ranked_F)

###################### USING Linear Regression of 3rd degree ###############

#Creating a model
model = LinearRegression().fit(RF_T, Y)

# Getting  coeff. and intercept
r_sq = model.score(RF_T, Y)
intercept, coefficients = model.intercept_, model.coef_
print(intercept,coefficients)

# Predicting the Water Saturation
SW_PREDICTED = model.predict(RF_T)
print(SW_PREDICTED)
print(Y)


########################## USING DIFFERENT MODEL########################

import statsmodels.api as sm
#x.astype(np.float)
Ranked_F=Ranked_F.astype(np.float)      #converting into float
Y=Y.astype(np.float) 
RF_C = sm.add_constant(Ranked_F)
modelA = sm.OLS(Y, Ranked_F)
results = modelA.fit()
print(results.summary())

'''
                                 OLS Regression Results                                
=======================================================================================
Dep. Variable:                      y   R-squared (uncentered):                   1.000
Model:                            OLS   Adj. R-squared (uncentered):              1.000
Method:                 Least Squares   F-statistic:                          2.065e+09
Date:                Tue, 28 Jul 2020   Prob (F-statistic):                        0.00
Time:                        16:18:42   Log-Likelihood:                          683.02
No. Observations:                3162   AIC:                                     -1360.
Df Residuals:                    3159   BIC:                                     -1342.
Df Model:                           3                                                  
Covariance Type:            nonrobust                                                  
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
x1             0.3513      0.003    117.223      0.000       0.345       0.357
x2             1.4238      0.055     25.996      0.000       1.316       1.531
x3            -0.7751      0.057    -13.499      0.000      -0.888      -0.663
==============================================================================
Omnibus:                      543.157   Durbin-Watson:                   0.122
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              856.673
Skew:                          -1.227   Prob(JB):                    9.46e-187
Kurtosis:                       3.694   Cond. No.                     1.08e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 1.08e+04. This might indicate that there are
strong multicollinearity or other numerical problems.
'''


print(results.predict())


















