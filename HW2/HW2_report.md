张悦枫

W15194175

#####Question 3

```python

import numpy as np

sig = 1

n = 200
T = 2 * n   # observation number
B = 1000   # simulation number

j_r = np.zeros(B)

for i in range(B):
    e = np.random.randn(T)
    e_t1 = np.zeros(T)
    for j in range(T-1):
        e_t1[j] = e[j+1]
    sig_c = (1/(4*n-2)) * sum(np.power(e[:T-1] + e_t1[:T-1],2))
    j_r[i] = 2 * np.sqrt(2*n - 1)*(sig_c - sig)
    
print("variance:", np.var(j_r))
print("mean:", j_r.mean())
```

Result:

>variance: 12.1358191575
>mean: 0.0471400754971

##### Question 4

```python
VR.py
import numpy as np                    # Python scientific computing package
import pandas as pd                   # Python data analysis package
from pandas_datareader import data    # data fetching utility
import matplotlib.pyplot as plt       # plot utility
plt.style.use('ggplot')
 
from VR_test import*         # self-defined VR test functions
 
fetch_data = 0                       # whether download data from Yahoo
 
if fetch_data == 1:
    tickers = ['^GSPC']
    data_source = 'yahoo'
    start_date = '2000-01-01'
    end_date = '2016-12-31'
    
    # get data
    raw_data = data.DataReader(tickers, data_source, start_date, end_date)
    sp_data = raw_data.minor_xs('^GSPC')
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
```

```python
VR_test.py
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
```

Result:

>The p value of J_r(3) statitic is 0.263658979266
>The null is not rejected at the 5% confidence level 
>
>The p value of J_r(5) statitic is 0.00894290248488
>The null is rejected at the 5% confidence level 
>
>The p value of J_r(10) statitic is 0.213315506174
>The null is not rejected at the 5% confidence level





