#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:03:46 2018

@author: marywang
"""
from matplotlib import pyplot as plt
import pandas as pd
import sklearn as skln
import numpy as np
import seaborn as sns
#from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pylab import scatter, show, legend, xlabel, ylabel
import statsmodels.api as sm
from scipy import stats
from statsmodels.sandbox.regression.predstd import wls_prediction_std

from sklearn.metrics import roc_curve

# read cvs file
filename = "VUMC_Schedule.csv"
names = ['SurgDate', 'DOW', 'T - 28', 'T - 21', 'T - 14', 'T - 13', 'T - 12', 'T - 11', 'T - 10',
         'T - 9', 'T - 8', 'T - 7', 'T - 6', 'T - 5', 'T - 4', 'T - 3', 'T - 2', 'T - 1', 'Actual'] 
VUMC = pd.read_csv(filename, names=names,dtype='object', encoding='utf-8', engine='c')
# get pure data
data = VUMC.drop(VUMC.index[0]) # drpp row 0: names
data = data.apply(pd.to_numeric, errors='ignore')
data[names] = data[names].apply(pd.to_numeric, errors='ignore')
#process DOW to dummy binary data Step 1: convert to bool type

#---------------------EDA Start-----------------------
#data.corr()
#sns.heatmap(data.corr())
#plt.title('Correlation Coefficient of all variables')
#plt.show()
#pd.read_csv(filename).corr()
T_X_mean = pd.read_csv(filename).mean(axis=0)
xais = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
scatter(xais, T_X_mean)
plt.title('Mean Number from T - 28 to Actual')
plt.ylabel('Mean Number')
plt.xlabel('Ranged Date')
show()
#---------------------EDA End-----------------------

data['isMon'] = data['T - 1']
data['isMon'] = data.DOW == 'Mon'
data['isTue'] = data.DOW == 'Tue'
data['isWed'] = data.DOW == 'Wed'
data['isThu'] = data.DOW == 'Thu'
data['isFri'] = data.DOW == 'Fri'
# Step 2: convert to int type
data['isMon'] = data['isMon'] * 1
data['isTue'] = data['isTue'] * 1
data['isWed'] = data['isWed'] * 1
data['isThu'] = data['isThu'] * 1
data['isFri'] = data['isFri'] * 1

X = data
y = data['Actual']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.24)

#-----------important!!!!!!!!!!!!!!-----
#data = X_train
#data_validation = X_validation
#data_test = X_test
#X_train.to_csv('X_train.csv')
#X_validation.to_csv('X_validation.csv')
#X_test.to_csv('X_test.csv')
#y_train.to_csv('y_train.csv')
#y_validation.to_csv('y_validation.csv')
#y_test.to_csv('y_test.csv')

#-----------important!!!!!!!!!!!!!!-----
X_train = pd.read_csv('X_train.csv')
X_validation = pd.read_csv('X_validation.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_validation = pd.read_csv('y_validation.csv')
y_test = pd.read_csv('y_test.csv')
#-----------important!!!!!!!!!!!!!!-----

#data = X_train

#-----------important!!!!!!!!!!!!!!-----
print(data.shape)
T_X1 = 'T - 1'
T_X2 = 'T - 2'
T_X   = 'T - 7'
isMon = 'isMon'
isTue = 'isTue'
isWed = 'isWed'
isThu = 'isThu'
isFri = 'isFri'

#---------------------EDA Start-----------------------
#------------------------------------plot DOW lines----------------------------
groups_DOW = data.groupby(['DOW'])
plt.plot(data['Actual'],linewidth = 0.4)
scatter(groups_DOW.get_group('Mon')['Actual'].index.tolist(), groups_DOW.get_group('Mon')['Actual'], s = 8, color = 'green')
scatter(groups_DOW.get_group('Tue')['Actual'].index.tolist(), groups_DOW.get_group('Tue')['Actual'], s = 8, color = 'red')
scatter(groups_DOW.get_group('Wed')['Actual'].index.tolist(), groups_DOW.get_group('Wed')['Actual'], s = 8, color = 'blue')
scatter(groups_DOW.get_group('Thu')['Actual'].index.tolist(), groups_DOW.get_group('Thu')['Actual'], s = 8, color = 'black')
scatter(groups_DOW.get_group('Fri')['Actual'].index.tolist(), groups_DOW.get_group('Fri')['Actual'], s = 8, color = 'darkviolet')
legend(['Actual','Mon','Tue','Wed','Thu','Fri'])
plt.title('Actual Number for Weekdays')
plt.ylabel('Actual Number')
plt.xlabel('Ranged Date')
show()

groups_DOW = pd.read_csv(filename).groupby(['DOW'])
xais = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
scatter(xais,groups_DOW.get_group('Mon').mean(),s = 15, color = 'green')
scatter(xais,groups_DOW.get_group('Tue').mean(),s = 15, color = 'red')
scatter(xais,groups_DOW.get_group('Wed').mean(),s = 15, color = 'blue')
scatter(xais,groups_DOW.get_group('Thu').mean(),s = 15, color = 'black')
scatter(xais,groups_DOW.get_group('Fri').mean(),s = 15,color = 'darkviolet')
legend(['Mon','Tue','Wed','Thu','Fri'])
plt.title('Mean Number for Weekdays')
plt.ylabel('Mean Number')
plt.xlabel('Date')
show()

# T - 7*DOW plot
LR1 = LinearRegression()
LR2 = LinearRegression()
LR3 = LinearRegression()
LR4 = LinearRegression()
LR5 = LinearRegression()
x1 = groups_DOW.get_group('Mon')['T - 7'].reshape(-1,1)
x2 = groups_DOW.get_group('Tue')['T - 7'].reshape(-1,1)
x3 = groups_DOW.get_group('Wed')['T - 7'].reshape(-1,1)
x4 = groups_DOW.get_group('Thu')['T - 7'].reshape(-1,1)
x5 = groups_DOW.get_group('Fri')['T - 7'].reshape(-1,1)
LR_T_7_1 = LR1.fit(x1,groups_DOW.get_group('Mon')['Actual'])
LR_T_7_2 = LR2.fit(x2,groups_DOW.get_group('Tue')['Actual'])
LR_T_7_3 = LR3.fit(x3,groups_DOW.get_group('Wed')['Actual'])
LR_T_7_4 = LR4.fit(x4,groups_DOW.get_group('Thu')['Actual'])
LR_T_7_5 = LR5.fit(x5,groups_DOW.get_group('Fri')['Actual'])

plt.plot(groups_DOW.get_group('Mon')['T - 7'], LR_T_7_1.predict(x1),linewidth=0.8, color = 'green')
plt.plot(groups_DOW.get_group('Tue')['T - 7'], LR_T_7_2.predict(x2),linewidth=0.8, color = 'red')
plt.plot(groups_DOW.get_group('Wed')['T - 7'], LR_T_7_3.predict(x3),linewidth=0.8, color = 'blue')
plt.plot(groups_DOW.get_group('Thu')['T - 7'], LR_T_7_4.predict(x4),linewidth=0.8, color = 'black')
plt.plot(groups_DOW.get_group('Fri')['T - 7'], LR_T_7_5.predict(x5),linewidth=0.8, color = 'darkviolet')
scatter(groups_DOW.get_group('Mon')['T - 7'],groups_DOW.get_group('Mon')['Actual'],s = 12, color = 'green')
scatter(groups_DOW.get_group('Tue')['T - 7'],groups_DOW.get_group('Tue')['Actual'],s = 12, color = 'red')
scatter(groups_DOW.get_group('Wed')['T - 7'],groups_DOW.get_group('Wed')['Actual'],s = 12, color = 'blue')
scatter(groups_DOW.get_group('Thu')['T - 7'],groups_DOW.get_group('Thu')['Actual'],s = 12, color = 'black')
scatter(groups_DOW.get_group('Fri')['T - 7'],groups_DOW.get_group('Fri')['Actual'],s = 12,color = 'darkviolet')
legend(['Mon','Tue','Wed','Thu','Fri'])
plt.title('T - 7*DOW Model')
plt.ylabel('Actual Number')
plt.xlabel('T - 7 Booking numbers')
plt.xlim(50,120)
plt.ylim(70,160)
show()
#---------------------EDA End-----------------------

#-----------------------model training: function def start---------------------
#Simple Linear Regression
def VUMC_SLR( T_X ):   
    print('---------------Start Simple Linear Regression as X = %s' % T_X)
    LR = LinearRegression()
#    
    X = data[T_X]
    Y = data['Actual']
    X = X.reshape(-1,1)
    LR_T_X = LR.fit(X, Y)
    #----plot-----
    plt.scatter(X, Y,  color='teal',s = 15)
    plt.plot(X, LR_T_X.predict(X), color='red', linewidth=1.5)
    plt.title('Linear Regression of %s' % T_X)
    plt.ylabel('Actual Number')
    plt.xlabel(T_X)
    plt.show()
    # The coefficients
    print('Coefficients: %f \nIntercept: %f '% (LR_T_X.coef_, LR_T_X.intercept_))
    #---Model Validation----:use MSE
    # The mean squared error
    print('---Model Validation----\nMSE: %.2f'
      % mean_squared_error(X_validation['Actual'], LR_T_X.predict(X_validation[T_X].reshape(-1,1))))
    print('---Model test----\nMSE: %.2f'
      % mean_squared_error(X_test['Actual'], LR_T_X.predict(X_test[T_X].reshape(-1,1))))
    print('Training score / R_Squared: %f' % LR_T_X.score(X.reshape(-1,1), Y))
    print('---------------End Simple Linear Regression as X = %s' % T_X)
    return;
    
#OLS
def VUMC_SLR_OLS( T_X ):   
    print('---------------Start VUMC_SLR_OLS Simple Linear Regression as X = %s' % T_X)
    x = data[T_X]
    y = data['Actual']
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, s = 5, alpha=0.5, color='teal')
    fig.suptitle('CI and PI of T - 7')
    
    x = x.reshape(-1,1)
    x = sm.add_constant(x)
    
    model = sm.OLS(y, x)
    fitted = model.fit()
#    x_pred = np.linspace(x.min(), x.max(), 50)
    x_pred = np.linspace(50, x.max(), 10)
    x_pred2 = sm.add_constant(x_pred)
    y_pred = fitted.predict(x_pred2)
    ax.plot(x_pred, y_pred, '-', color='black', linewidth=0.2)
    
    
    y_hat = fitted.predict(x) # x is an array from line 12 above
    y_err = y - y_hat
    mean_x = x.T[1].mean()
    n = len(x)
    dof = n - fitted.df_model - 1
    t = stats.t.ppf(1-0.01, df=dof)
    s_err = np.sum(np.power(y_err, 2))
    conf = t * np.sqrt((s_err/(n-2))*(1.0/n + 
        (np.power((x_pred-mean_x),2)/ ((np.sum(np.power(x_pred,2))) - n*(np.power(mean_x,2))))))
    upper = y_pred + abs(conf)
    lower = y_pred - abs(conf)
    ax.fill_between(x_pred, lower, upper, color='orchid', alpha=0.4)
    print("x_pred")
    print(x_pred)
    print('CI upper - lower: ')
    print((upper - lower).mean())
    print(upper)
    print(lower)
    
    sdev, lower, upper = wls_prediction_std(fitted, exog=x_pred2, alpha=0.01)
    ax.fill_between(x_pred, lower, upper, color='#888888', alpha=0.1)
    ax.legend(['Regression','Data','99% CI','99% PI'])
    print('PI upper - lower: ')
    print((upper - lower).mean())
    print(upper)
    print(lower)
    

    return;
    
#Mltiple Linear Regression
def VUMC_MLR( T_X1, T_X2 ): 
    print('---------------Start Multiple Linear Regression')
    X = data[[T_X1, T_X2]]
    #let OLS summary output contain const(intercept)
    X = sm.add_constant(X)
    Y = data['Actual']
    MLR = sm.OLS(Y, X).fit()
    X_validation_used = X_validation[[T_X1, T_X2]]
    X_validation_used = sm.add_constant(X_validation_used)
    X_test_used = X_test[[T_X1, T_X2]]
    X_test_used = sm.add_constant(X_test_used)
    print('---Model Validation----\nMSE: %.2f'
      % mean_squared_error(X_validation['Actual'], MLR.predict(X_validation_used) ))
    print('---Model test----\nMSE: %.2f'
      % mean_squared_error(X_test['Actual'], MLR.predict(X_test_used) ))
    print(MLR.summary())
    return;
    
#Mltiple Linear Regression contain ADD binary variable(DOW)
def VUMC_MLR_DOW( T_X, isMon, isTue, isWed, isThu, isFri ): 
    X = data[[T_X, isMon, isTue, isWed, isThu, isFri]]
    #let OLS summary output contain const(intercept)
    X = sm.add_constant(X)
    Y = data['Actual']
    MLR = sm.OLS(Y, X).fit()
#    Y_pred = MLR.predict(X)
    #---Model Validation----:use MSE
    X_validation_used = X_validation[[T_X, isMon, isTue, isWed, isThu, isFri]]
    X_validation_used = sm.add_constant(X_validation_used)
    X_test_used = X_test[[T_X, isMon, isTue, isWed, isThu, isFri]]
    X_test_used = sm.add_constant(X_test_used)
    print('---Model Validation----\nMSE: %.2f'
      % mean_squared_error(X_validation['Actual'], MLR.predict(X_validation_used) ))
    print('---Model test----\nMSE: %.2f'
      % mean_squared_error(X_test['Actual'], MLR.predict(X_test_used) ))
    print(MLR.summary())
    print('---------------End Multiple Linear Regression')
    return;
    
#Mltiple Linear Regression contain MULTIPLICATION binary variable(DOW)
def VUMC_MLR_MULTI_DOW( T_X, isMon, isTue, isWed, isThu, isFri ): 
    print('---------------Start VUMC_MLR_MULTI_DOW')
    X = pd.DataFrame(data[T_X] * data[isMon], columns = [T_X+'*isMon'])
    X[T_X+'*isTue'] = data[T_X] * data[isTue]
    X[T_X+'*isWed'] = data[T_X] * data[isWed]
    X[T_X+'*isThu'] = data[T_X] * data[isThu]
    X[T_X+'*isFri'] = data[T_X] * data[isFri]
    #let OLS summary output contain const(intercept)
    X = sm.add_constant(X)
    Y = data['Actual']
    MLR = sm.OLS(Y, X).fit()
#    Y_pred = MLR.predict(X)
     #---Model Validation----:use MSE
    X_validation_used = pd.DataFrame(X_validation[T_X] * X_validation[isMon], columns = [T_X+'*isMon'])
    X_validation_used[T_X+'*isTue'] = X_validation[T_X] * X_validation[isTue]
    X_validation_used[T_X+'*isWed'] = X_validation[T_X] * X_validation[isWed]
    X_validation_used[T_X+'*isThu'] = X_validation[T_X] * X_validation[isThu]
    X_validation_used[T_X+'*isFri'] = X_validation[T_X] * X_validation[isFri]
    X_validation_used = sm.add_constant(X_validation_used)
    
    X_test_used = pd.DataFrame(X_test[T_X] * X_test[isMon], columns = [T_X+'*isMon'])
    X_test_used[T_X+'*isTue'] = X_test[T_X] * X_test[isTue]
    X_test_used[T_X+'*isWed'] = X_test[T_X] * X_test[isWed]
    X_test_used[T_X+'*isThu'] = X_test[T_X] * X_test[isThu]
    X_test_used[T_X+'*isFri'] = X_test[T_X] * X_test[isFri]
    X_test_used = sm.add_constant(X_test_used)
    print('---Model Validation----\nMSE: %.2f'
      % mean_squared_error(X_validation['Actual'], MLR.predict(X_validation_used) ))
    print('---Model test----\nMSE: %.2f'
      % mean_squared_error(X_test['Actual'], MLR.predict(X_test_used) ))
    print(MLR.summary())
    print('---------------End VUMC_MLR_MULTI_DOW')
    return;
#---------------------model training: function def end-----------------------
#---------------------model training: start------------------------------
#------------------------------------VUMC_SLR---------------------------------
#VUMC_SLR( T_X = 'T - 1')
#VUMC_SLR( T_X = 'T - 2')
#VUMC_SLR( T_X = 'T - 3')
#VUMC_SLR( T_X = 'T - 4')
#VUMC_SLR( T_X = 'T - 5')
#VUMC_SLR( T_X = 'T - 6')
VUMC_SLR( T_X = 'T - 7')
#VUMC_SLR( T_X = 'T - 8')
#VUMC_SLR( T_X = 'T - 9')
#VUMC_SLR( T_X = 'T - 10')
#VUMC_SLR( T_X = 'T - 11')
#VUMC_SLR( T_X = 'T - 12')
#VUMC_SLR( T_X = 'T - 13')
#VUMC_SLR( T_X = 'T - 14')
#VUMC_SLR( T_X = 'T - 21')
#VUMC_SLR( T_X = 'T - 28')

#----------------------------VUMC_MLR---------------------------------
VUMC_MLR( T_X1 = 'T - 7', T_X2 = 'T - 8')
#VUMC_MLR( T_X1 = 'T - 7', T_X2 = 'T - 9')
#VUMC_MLR( T_X1 = 'T - 7', T_X2 = 'T - 10')
#VUMC_MLR( T_X1 = 'T - 7', T_X2 = 'T - 11')
#VUMC_MLR( T_X1 = 'T - 7', T_X2 = 'T - 12')
#VUMC_MLR( T_X1 = 'T - 7', T_X2 = 'T - 13')
#VUMC_MLR( T_X1 = 'T - 7', T_X2 = 'T - 14')
#VUMC_MLR( T_X1 = 'T - 7', T_X2 = 'T - 21')
#VUMC_MLR( T_X1 = 'T - 7', T_X2 = 'T - 28')

#VUMC_MLR( T_X1 = 'T - 8', T_X2 = 'T - 9')
#VUMC_MLR( T_X1 = 'T - 8', T_X2 = 'T - 10')
#VUMC_MLR( T_X1 = 'T - 8', T_X2 = 'T - 11')
#VUMC_MLR( T_X1 = 'T - 8', T_X2 = 'T - 12')
#VUMC_MLR( T_X1 = 'T - 8', T_X2 = 'T - 13')
#VUMC_MLR( T_X1 = 'T - 8', T_X2 = 'T - 14')
#VUMC_MLR( T_X1 = 'T - 8', T_X2 = 'T - 21')
#VUMC_MLR( T_X1 = 'T - 8', T_X2 = 'T - 28')

#VUMC_MLR( T_X1 = 'T - 9', T_X2 = 'T - 10')
#VUMC_MLR( T_X1 = 'T - 9', T_X2 = 'T - 11')
#VUMC_MLR( T_X1 = 'T - 9', T_X2 = 'T - 12')
#VUMC_MLR( T_X1 = 'T - 9', T_X2 = 'T - 13')
#VUMC_MLR( T_X1 = 'T - 9', T_X2 = 'T - 14')
#VUMC_MLR( T_X1 = 'T - 9', T_X2 = 'T - 21')
#VUMC_MLR( T_X1 = 'T - 9', T_X2 = 'T - 28')


#VUMC_MLR( T_X1 = 'T - 10', T_X2 = 'T - 11')
#VUMC_MLR( T_X1 = 'T - 10', T_X2 = 'T - 12')
#VUMC_MLR( T_X1 = 'T - 10', T_X2 = 'T - 13')
#VUMC_MLR( T_X1 = 'T - 10', T_X2 = 'T - 14')
#VUMC_MLR( T_X1 = 'T - 10', T_X2 = 'T - 21')
#VUMC_MLR( T_X1 = 'T - 10', T_X2 = 'T - 28')
#
#
#VUMC_MLR( T_X1 = 'T - 11', T_X2 = 'T - 12')
#VUMC_MLR( T_X1 = 'T - 11', T_X2 = 'T - 13')
#VUMC_MLR( T_X1 = 'T - 11', T_X2 = 'T - 14')
#VUMC_MLR( T_X1 = 'T - 11', T_X2 = 'T - 21')
#VUMC_MLR( T_X1 = 'T - 11', T_X2 = 'T - 28')
#
#
#VUMC_MLR( T_X1 = 'T - 12', T_X2 = 'T - 13')
#VUMC_MLR( T_X1 = 'T - 12', T_X2 = 'T - 14')
#VUMC_MLR( T_X1 = 'T - 12', T_X2 = 'T - 21')
#VUMC_MLR( T_X1 = 'T - 12', T_X2 = 'T - 28')

#
#VUMC_MLR( T_X1 = 'T - 13', T_X2 = 'T - 14')
#VUMC_MLR( T_X1 = 'T - 13', T_X2 = 'T - 21')
#VUMC_MLR( T_X1 = 'T - 13', T_X2 = 'T - 28')
#
#VUMC_MLR( T_X1 = 'T - 14', T_X2 = 'T - 21')
#VUMC_MLR( T_X1 = 'T - 14', T_X2 = 'T - 28')
#
#
#VUMC_MLR( T_X1 = 'T - 21', T_X2 = 'T - 28')


#----------------------------VUMC_SLR_DOW---------------------------------
#VUMC_MLR_DOW( 'T - 1', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 2', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 3', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 4', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 5', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 6', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
VUMC_MLR_DOW( 'T - 7', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 8', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 9', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 10', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 11', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 12', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 13', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 14', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 21', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_DOW( 'T - 28', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')

#----------------------------VUMC_MLR_MULTI_DOW---------------------------
#VUMC_MLR_MULTI_DOW( 'T - 1', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 2', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 3', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 4', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 5', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 6', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
VUMC_MLR_MULTI_DOW( 'T - 7', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 8', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 9', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 10', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 11', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 12', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 13', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 14', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 21', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')
#VUMC_MLR_MULTI_DOW( 'T - 28', 'isMon', 'isTue', 'isWed', 'isThu', 'isFri')

print('Successsssssssssssssssssssssss')
