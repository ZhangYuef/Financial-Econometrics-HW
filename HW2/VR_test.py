#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 13:30:50 2017

@author: zyf
"""
import scipy.stats as st                 # Python statistical tools
import numpy as np
import math
 
def J_r_q_stat(data,q):
    '''
    ret 1:  float 
            J_r_q statistic
    ret 2:  int
            sample size devided by q
    '''
    residue = len(data) % q
   
    data = data[residue:]

    n = len(data)/q
    
    sigma_a2 = sum(np.power(data, 2))
    
    tSum = []
    
    for i in range(1, int(n)):
        tData = data[q*(i-1) : q*i]
        tSum.append(np.sum(tData))
    
    sigma_b2 = sum(np.power(tSum,2))
    
    return sigma_b2 / sigma_a2 - 1, n
 
 
def VR_test(data,q):
    # get the J_r_q statistic
    J_r_q, n = J_r_q_stat(data,q)
    # calculate the p-value (two-tail test)
    return (1-st.norm.cdf(math.sqrt(n)*abs(J_r_q),0,math.sqrt(2*(q-1)/q)))*2