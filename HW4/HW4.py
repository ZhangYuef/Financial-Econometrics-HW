#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 13:04:15 2017

@author: zyf
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
 
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tsplot
 
data = pd.read_csv('/Users/zyf/Projects/Financial-Econometrics-HW/HW4/f_curve.csv')

def forwarding_rate(t,tao):
    return data.loc[t,'BETA0']+\
    data.loc[t,'BETA1']*math.exp(-tao/ data.loc[t,'GAMMA1']) +\
    data.loc[t,'BETA2']*tao / data.loc[t,'GAMMA1'] * math.exp(-tao/ data.loc[t,'GAMMA1']) +\
    data.loc[t,'BETA3']*tao / data.loc[t,'GAMMA2'] * math.exp(-tao/ data.loc[t,'GAMMA2'])

def f_r_array(tao):
    # f_r(tao) for Jan 2010 to Oct 2017
    tmp = np.zeros(94)
    pst = 1977 #2010-1-4
    for i in range(94) :
        tmp[i] = forwarding_rate(pst,tao)
        pst = pst -20
    return tmp

tao_l = [1/4,1,3,5,7,10]

f_r = np.empty([1,94,6])
index = 0
for i in tao_l:
    f_r[:,:,index] = f_r_array(i)
    index = index+1

def f_r_array_past(tao):
    # from Jan 1980 to Dec 2009
    tmp = np.zeros(360)
    pst = 9457 #1980-1-2
    for i in range(360) :
        tmp[i] = forwarding_rate(pst,tao)
        pst = pst -20
    return tmp

#past[] from Jan 1980 to Dec 2009,360 months in total
f_r_past = np.empty([1,360,6])
index = 0
for i in tao_l:
    f_r_past[:,:,index] = f_r_array_past(i)
    index = index+1

ret = np.diff(f_r_past[:,:,0])
ret = np.reshape(ret,[359,1])
T = len(ret)
'''
print(len(ret))
plt.figure(1)
plt.plot(ret)
'''
ARMR_mod = sm.tsa.ARMA(ret,order=(1,0))
ARMR_res = ARMR_mod.fit()
print(ARMR_res.summary())
#prediction
plt.figure(1)
ARMR_res.plot_predict(start=T-12,end=T+12)















