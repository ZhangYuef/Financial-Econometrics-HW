# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 16:11:08 2017

@author: zyf
"""

import numpy as np                    # Python scientific computing package
import pandas as pd                   # Python data analysis package
from pandas_datareader import data    # data fetching utility
import matplotlib.pyplot as plt       # plot utility
plt.style.use('ggplot')
 
from VR_test import*         # self-defined VR test functions
 
fetch_data = 0                       # whether download data from Yahoo
 
if fetch_data == 1:
    # stock short code (SP500)
    tickers = ['^GSPC']
    
    # obtain data from Yahoo Finance
    data_source = 'yahoo'
    
    # data range
    start_date = '2000-01-01'
    end_date = '2016-12-31'
    
    # get data
    raw_data = data.DataReader(tickers, data_source, start_date, end_date)
    
    # keep only SP500 data
    sp_data = raw_data.minor_xs('^GSPC')
    
    # save data to a file for future use
    sp_data.to_pickle('sp_data.pkl')
else:
    sp_data = pd.read_pickle('sp_data.pkl')
 
# resample to get weekly data
sp_weekly = sp_data.resample('W').last()
 
# get the log returns
log_return = np.log(sp_weekly).diff()[1:]
 
# save the log returns
log_return.to_pickle('log_return_data.pkl')
 
 
# conduct the basic Variance Ratio test
#J_r_3 statistic
p_val = VR_test(log_return['Adj Close'].values,3)
# print the p value of the test as well as the result of the test
print('The p value of J_r(3) statitic is', p_val)
if p_val <= 0.05:
    print('The null is rejected at the 5% confidence level')
else:
    print('The null is not rejected at the 5% confidence level \n')
    
#J_r_5 statistic
p_val = VR_test(log_return['Adj Close'].values,5)
# print the p value of the test as well as the result of the test
print('The p value of J_r(5) statitic is', p_val)
if p_val <= 0.05:
    print('The null is rejected at the 5% confidence level \n')
else:
    print('The null is not rejected at the 5% confidence level')
    
#J_r_10 statistic
p_val = VR_test(log_return['Adj Close'].values,10)
# print the p value of the test as well as the result of the test
print('The p value of J_r(10) statitic is', p_val)
if p_val <= 0.05:
    print('The null is rejected at the 5% confidence level')
else:
    print('The null is not rejected at the 5% confidence level')
    
    
