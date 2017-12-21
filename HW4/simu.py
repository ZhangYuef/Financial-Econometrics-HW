#! py
import xlrd as xl
import numpy as np
import pandas as pd
from datetime import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def parse_ymd(s):
    year_s, mon_s, day_s = s.split('-')
    return datetime(int(year_s), int(mon_s), int(day_s))

def getNextPredictByArmaModel(inData, modelOrder):
    try:
        getTheModel = sm.tsa.ARMA(inData, order=modelOrder).fit(disp=0)
        return getTheModel.predict(start=len(inData), end=len(inData))[0]
    except:
        return 'NAN'

def getNextPredictCollection(modelOrder, theTau):
    resList = np.zeros(totPred)
    for index in range(totPred):
        resList[index] = getNextPredictByArmaModel(
            data.values[:(index + indexBegin), theTau], modelOrder=modelOrder)
    return resList


def getNextPredictVAR(modelOrder):
    resList = np.zeros((totPred, 6))
    for index in range(totPred):
        usedData = data.values[:(index + indexBegin)]
        resList[index] = sm.tsa.VAR(usedData).fit(
            modelOrder).forecast(usedData[-modelOrder:], 1)
    return resList


def getNextPredictAVE(theTau):
    resList = np.zeros((totPred))
    for index in range(totPred):
        resList[index] = 1 / (index + indexBegin) * \
            sum(data.values[:(index + indexBegin), theTau])
    return resList


def getNextPredictLAST(theTau):
    resList = np.zeros((totPred))
    for index in range(totPred):
        usedData = data.values[:(index + indexBegin), theTau]
        resList[index] = usedData[-1]
    return resList


def getMseValue(estimateResult, theTau):
    realData = data.values[indexBegin:-1, theTau]
    tmpList = estimateResult - realData
    return 1 / totPred * sum(np.power(tmpList, 2))


coef = xl.open_workbook("./f_curve.xlsx").sheets()[0]
totM = coef.nrows - 1
fList = []
dateList = []
tau = np.array([1 / 4, 1, 3, 5, 7, 10])
for index  in range(totM):
    date = coef.cell(1 + index, 0).value
    date = parse_ymd(date)
    beta0 = coef.cell(1 + index, 1).value
    beta1 = coef.cell(1 + index, 2).value
    beta2 = coef.cell(1 + index, 3).value
    beta3 = coef.cell(1 + index, 4).value
    gamma1 = coef.cell(1 + index, 5).value
    gamma2 = coef.cell(1 + index, 6).value
    # fValue 6x1
    fValue = beta0 + beta1 * np.exp(-tau / gamma1) +\
            beta2 * (tau / gamma1) *\
            np.exp(-tau / gamma1) +\
            beta3 * (tau / gamma2) * np.exp(-tau / gamma2)
    dateList.append(date)
    fList.append(fValue.tolist())
    
data = pd.DataFrame(fList, index=dateList, columns=[
    '1/4', '1', '3', '5', '7', '10']).resample('M').last()

totPred = len(data['2010-01':'2017-10'])
indexBegin = len(data['1980-01':'2009-12'])

mseValue = np.zeros(6)
for index in range(6):
    mseValue[index] = getMseValue(getNextPredictCollection((1, 0), index), index)
print('the AR(1) \'s mse list is:\n')
print(mseValue)
print('\n--------------------------------------\n')
for index in range(6):
    mseValue[index] = getMseValue(getNextPredictCollection((2, 0), index), index)
print('the AR(2) \'s mse list is:\n')
print(mseValue)
print('\n--------------------------------------\n')
for index in range(6):
    mseValue[index] = getMseValue(getNextPredictCollection((0, 1), index), index)
print('the MA(1) \'s mse list is:\n')
print(mseValue)
print('\n--------------------------------------\n')
for index in range(6):
    mseValue[index] = getMseValue(getNextPredictCollection((0, 2), index), index)
print('the MA(2) \'s mse list is:\n')
print(mseValue)
print('\n--------------------------------------\n')
for index in range(6):
    mseValue[index] = getMseValue(getNextPredictVAR(1)[:, index], index)
print('the VAR(1) \'s mse list is:\n')
print(mseValue)
print('\n--------------------------------------\n')
for index in range(6):
    mseValue[index] = getMseValue(getNextPredictVAR(2)[:, index], index)
print('the VAR(2) \'s mse list is:\n')
print(mseValue)
print('\n--------------------------------------\n')
for index in range(6):
    mseValue[index] = getMseValue(getNextPredictAVE(index), index)
print('the AVE \'s mse list is:\n')
print(mseValue)
print('\n--------------------------------------\n')
for index in range(6):
    mseValue[index] = getMseValue(getNextPredictLAST(index), index)
print('the LAST \'s mse list is:\n')
print(mseValue)
print('\n--------------------------------------\n')
