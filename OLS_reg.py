#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:41:07 2020

@author: mvr
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 16:29:21 2020

@author: mvr
"""
############# BUILDING A MODEL USING BVW,TEMP,PHIF #################
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression,LinearRegression
import sklearn

array = df.values
X1 = array[:,1:2]
X2=array[:,6:7]
X3=array[:,12:13]
X=np.concatenate((X1,X2),axis=1)
X_BVW_TEMP_PHIF=np.concatenate((X,X3),axis=1)
Y = array[:,11:12]


import statsmodels.api as sm
#x.astype(np.float)
X_BVW_TEMP_PHIF=X_BVW_TEMP_PHIF.astype(np.float)      #converting into float
Y=Y.astype(np.float) 
X_BVW_TEMP_PHIF = sm.add_constant(X_BVW_TEMP_PHIF)
modelA = sm.OLS(Y, X_BVW_TEMP_PHIF)
results = modelA.fit()
print(results.summary())
print(results.predict())

'''#USING PHIF,RHOMA,RW
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


'''#USING BVW,TEMP,PHIF

                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       1.000
Model:                            OLS   Adj. R-squared:                  1.000
Method:                 Least Squares   F-statistic:                 1.618e+10
Date:                Tue, 28 Jul 2020   Prob (F-statistic):               0.00
Time:                        18:28:04   Log-Likelihood:                 4059.0
No. Observations:                3162   AIC:                            -8110.
Df Residuals:                    3158   BIC:                            -8086.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.7756      0.059     13.165      0.000       0.660       0.891
x1             5.5740      0.037    149.685      0.000       5.501       5.647
x2            -4.5738      0.037   -123.796      0.000      -4.646      -4.501
x3             0.0006      0.000      1.181      0.238      -0.000       0.002
==============================================================================
Omnibus:                      285.842   Durbin-Watson:                   0.336
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              389.857
Skew:                           0.739   Prob(JB):                     2.21e-85
Kurtosis:                       3.880   Cond. No.                     2.78e+04
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 2.78e+04. This might indicate that there are
strong multicollinearity or other numerical problems.

print(results.predict())
[ 8.764e-01  8.747e-01  8.526e-01 ... -9.992e+02 -9.992e+02 -9.992e+02]



'''
