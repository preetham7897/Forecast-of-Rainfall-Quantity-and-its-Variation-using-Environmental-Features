import pandas as pd
from sklearn.model_selection import RepeatedKFold as rkf
from sklearn import linear_model as lm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score as r2
import numpy as np
from sklearn.utils import shuffle

def rmse(y_t, y_p):
    return (mse(y_t, y_p))**0.5

data_list = ['All Districts', 'Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Ariyalur','Chennai','Coimbatore','Cuddalore','Dharmapuri','Dindigul','Erode','Kancheepuram','Karur','Madurai','Nagapattinam','Namakkal','Perambalur','Pudukkottai','Ramanathapuram','Salem','Sivaganga','Thanjavur','Theni','The Nilgiris','Thiruvallur','Thiruvarur','Thoothukkudi','Tiruchirapalli','Tirunelveli','Tiruvannamalai','Vellore','Viluppuram','Virudhunagar']
parameters = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Parameters\\Parameters.csv')
rkf = rkf(n_splits=10, n_repeats=10)
#columns for result
dl = []
m = []
mse_ts = []
rmse_ts = []
mae_ts = []
mdae_ts = []
evs_ts = []
r2_ts = []
#iterating through datas
method = ['Multiple Linear Regression', 'Support Vector Regression', 'Decision Tree Regression', 'Polynomial Regression']
for i in data_list:
    param = parameters[parameters['Data'] == i]
    data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\' + i + '.csv')
    data = shuffle(data)
    #per data result values
    meth = []
    mse_m = []
    rmse_m = []
    mae_m = []
    mdae_m = []
    evs_m = []
    r2_m = []
    #Parameter Values
    k = list(param['SVR Kernel'])[0]
    md = list(param['DTR Max Depth'])[0]
    deg = list(param['PR Degree'])[0]
    #Creating models
    mlr = lm.LinearRegression()
    svr = SVR(kernel=k, epsilon=0.1, C=1)
    dt = dtr(max_depth=md)
    poly = pf(degree=deg)
    pr = lm.LinearRegression()
    c = 0
    #Repeated K Fold Cross Validation
    for tr_i, ts_i in rkf.split(data):
        print(i, c)
        train, test = data.iloc[tr_i], data.iloc[ts_i]
        train_x = train.drop(columns=['Index', 'District', 'Rainfall'])
        train_y = train['Rainfall']
        test_x = test.drop(columns=['Index', 'District', 'Rainfall'])
        test_y = test['Rainfall']
        poly_tr = poly.fit_transform(train_x)
        poly_ts = poly.fit_transform(test_x)
        #Fitting the data in the model
        mlr.fit(train_x, train_y)
        svr.fit(train_x, train_y)
        dt.fit(train_x, train_y)
        pr.fit(poly_tr, train_y)
        #Predicting the test
        mlr_p = mlr.predict(test_x)
        svr_p = svr.predict(test_x)
        dt_p = dt.predict(test_x)
        pr_p = pr.predict(poly_ts)
        #Methods
        meth.append(method[0])
        meth.append(method[1])
        meth.append(method[2])
        meth.append(method[3])
        #MSE for models
        mse_m.append(mse(test_y, mlr_p))
        mse_m.append(mse(test_y, svr_p))
        mse_m.append(mse(test_y, dt_p))
        mse_m.append(mse(test_y, pr_p))
        #RMSE for models
        rmse_m.append(rmse(test_y, mlr_p))
        rmse_m.append(rmse(test_y, svr_p))
        rmse_m.append(rmse(test_y, dt_p))
        rmse_m.append(rmse(test_y, pr_p))
        #MAE for models
        mae_m.append(mae(test_y, mlr_p))
        mae_m.append(mae(test_y, svr_p))
        mae_m.append(mae(test_y, dt_p))
        mae_m.append(mae(test_y, pr_p))
        #MDAE for models
        mdae_m.append(mdae(test_y, mlr_p))
        mdae_m.append(mdae(test_y, svr_p))
        mdae_m.append(mdae(test_y, dt_p))
        mdae_m.append(mdae(test_y, pr_p))
        #EVS for models
        evs_m.append(evs(test_y, mlr_p))
        evs_m.append(evs(test_y, svr_p))
        evs_m.append(evs(test_y, dt_p))
        evs_m.append(evs(test_y, pr_p))
        #R2 for models
        r2_m.append(r2(test_y, mlr_p))
        r2_m.append(r2(test_y, svr_p))
        r2_m.append(r2(test_y, dt_p))
        r2_m.append(r2(test_y, pr_p))
        c += 1
    #Converting the results to dict to dataframe
    d = {}
    d['Method'] = meth
    d['MSE'] = mse_m
    d['RMSE'] = rmse_m
    d['MAE'] = mae_m
    d['MDAE'] = mdae_m
    d['EVS'] = evs_m
    d['R2'] = r2_m
    df = pd.DataFrame(d, columns=['Method', 'MSE', 'RMSE', 'MAE', 'MDAE', 'EVS', 'R2'])
    #Mean value for the models
    for j in method:
        print(i, j)
        d = df[df['Method'] == j]
        dl.append(i)
        m.append(j)
        mse_ts.append(np.mean(list(d['MSE'])))
        rmse_ts.append(np.mean(list(d['RMSE'])))
        mae_ts.append(np.mean(list(d['MAE'])))
        mdae_ts.append(np.mean(list(d['MDAE'])))
        evs_ts.append(np.mean(list(d['EVS'])))
        r2_ts.append(np.mean(list(d['R2'])))
d = {}
d['Data'] = dl
d['Method'] = m
d['MSE'] = mse_ts
d['RMSE'] = rmse_ts
d['MAE'] = mae_ts
d['MDAE'] = mdae_ts
d['EVS'] = evs_ts
d['R2'] = r2_ts
df = pd.DataFrame(d, columns=['Data', 'Method', 'MSE', 'RMSE', 'MAE', 'MDAE', 'EVS', 'R2'])
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Best Model per Data\\Main Results.csv', index=False)