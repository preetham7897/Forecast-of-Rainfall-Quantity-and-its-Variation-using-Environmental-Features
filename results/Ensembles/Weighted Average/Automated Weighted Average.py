import pandas as pd
from itertools import combinations
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Ensembles\\Main - Adaboost - Chennai.csv')
models = list(data['Models'].unique())
print(data)
models = [LinearRegression(), DecisionTreeRegressor(max_depth=6), LinearRegression(), SVR(kernel='linear')]
names = ['MLR', 'DTR(6)', 'PR(4)', 'SVR(L)']
comb_models = []
comb_names = []
for i in range(1, len(models)+1):
    l = combinations(models, i)
    m = combinations(names, i)
    for j in l:
        comb_models.append(list(j))
    for j in m:
        comb_names.append(list(j))
print(comb_names)
print(len(comb_names))
"""
data = data[data['Loss'] == 'linear']
errors = ['MSE', 'RMSE', 'MAE', 'MDAE']
d = {}
for i in errors:
    l = list(data[i])
    m = []
    for j in l:
        m.append(1-j)
    d[i] = m
d['Models'] = models
d['EVS'] = data['EVS']
d['R2'] = data['R2']
df = pd.DataFrame(d, columns=['Models', 'MSE', 'RMSE', 'MAE', 'MDAE', 'EVS', 'R2'])
print(df)
mse_f = []
rmse_f = []
mae_f = []
mdae_f = []
evs_f = []
r2_f = []
errors = ['MSE', 'RMSE', 'MAE', 'MDAE', 'EVS', 'R2']
di = {}
for i in comb_models:
    d = {}
    df1 = df[df['Models'].isin(i)]
    for j in errors:
        l = list(df1[j])
        m = []
        for k in l:
            m.append(k/np.sum(l))
        d[j] = m
    d['Models'] = i
    d = pd.DataFrame(d, columns=['Models', 'MSE', 'RMSE', 'MAE', 'MDAE', 'EVS', 'R2'])
    print(i)
    m = []
    for j in i:
        l = []
        df1 = d[d['Models'] == j]
        for k in errors:
            l.append(list(df1[k]))
        m.append(np.mean(l))
    df1 = data[data['Models'].isin(i)]
    n = []
    for j in errors:
        l = list(df1[j])
        o = []
        for k, p in zip(l, m):
            o.append(k*p)
        print(np.mean(o))
        print(np.mean(l))
    print()for i in comb_models:
    print(i)
    df1 = df[df['Models'].isin(i)]
    for j in errors:
        
        for k in """