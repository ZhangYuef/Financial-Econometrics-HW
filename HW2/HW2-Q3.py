
# -*- coding: utf-8 -*-

"""
2017-10-24 15:38:42

@Zhang Yuefeng
"""

import numpy as np

import matplotlib.pyplot as plt       # plot utility
plt.style.use('ggplot')
 

sig = 1

n = 200
T = 2 * n   # observation number
B = 10000   # simulation number

j_r = np.zeros(B)

for i in range(B):
    e = np.random.randn(T)
    e_t1 = np.zeros(T)
    for j in range(0,T-1):
        e_t1[j] = e[j+1]
    sig_c = (1/(4*n-2)) * sum(np.power(e[:T-1] + e_t1[:T-1],2))
    j_r[i] = 2 * np.sqrt(2*n - 1)*(sig_c - sig)
    
print("variance:", np.var(j_r))
print("mean:", j_r.mean())

plt.figure(1)
plt.plot(j_r)