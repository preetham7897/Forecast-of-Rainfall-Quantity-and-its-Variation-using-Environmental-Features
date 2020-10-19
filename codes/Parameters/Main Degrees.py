import pandas as pd
from sklearn.model_selection import RepeatedKFold as rkf
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn import linear_model as lm
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
deg = [None, 2, 3, 4, 5]
#columns for result
dl = []
degree = []
mse_ts = []
rmse_ts = []
mae_ts = []
mdae_ts = []
evs_ts = []
r2_ts = []
rkf = rkf(n_splits=10, n_repeats=10)
pr = lm.LinearRegression()
for i in data_list:
    data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\' + i + '.csv')
    data = shuffle(data)
    #Error measures init
    d = []
    mse_d = []
    rmse_d = []
    mae_d = []
    mdae_d = []
    evs_d = []
    r2_d = []
    c = 0
    #Repeated K Fold Cross Validation
    for tr_i, ts_i in rkf.split(data):
        train, test = data.iloc[tr_i], data.iloc[ts_i]
        train_x = train.drop(columns=['District', 'Index', 'Rainfall'])
        train_y = train['Rainfall']
        test_x = test.drop(columns=['District', 'Index', 'Rainfall'])
        test_y = test['Rainfall']
        for j in deg:
            print(i, c, j)
            poly = pf(degree=j)
            poly_tr = poly.fit_transform(train_x)
            poly_ts = poly.fit_transform(test_x)
            pr.fit(poly_tr, train_y)
            pr_p = pr.predict(poly_ts)
            #Error values
            d.append(j)
            mse_d.append(mse(test_y, pr_p))
            rmse_d.append(rmse(test_y, pr_p))
            mae_d.append(mae(test_y, pr_p))
            mdae_d.append(mdae(test_y, pr_p))
            evs_d.append(evs(test_y, pr_p))
            r2_d.append(r2(test_y, pr_p))
        c += 1
    t = {}
    t['Degree'] = d
    t['MSE'] = mse_d
    t['RMSE'] = rmse_d
    t['MAE'] = mae_d
    t['MDAE'] = mdae_d
    t['EVS'] = evs_d
    t['R2'] = r2_d
    tf = pd.DataFrame(t, columns=['Degree', 'MSE', 'RMSE', 'MAE', 'MDAE', 'EVS', 'R2'])
    for j in deg:
        temp = tf[tf['Degree'] == j]
        dl.append(i)
        degree.append(j)
        mse_ts.append(np.mean(list(temp['MSE'])))
        rmse_ts.append(np.mean(list(temp['RMSE'])))
        mae_ts.append(np.mean(list(temp['MAE'])))
        mdae_ts.append(np.mean(list(temp['MDAE'])))
        evs_ts.append(np.mean(list(temp['EVS'])))
        r2_ts.append(np.mean(list(temp['R2'])))
d = {}
d['Data'] = dl
d['Degree'] = degree
d['MSE'] = mse_ts
d['RMSE'] = rmse_ts
d['MAE'] = mae_ts
d['MDAE'] = mdae_ts
d['EVS'] = evs_ts
d['R2'] = r2_ts
df = pd.DataFrame(d, columns=['Data', 'Degree', 'MSE', 'RMSE', 'MAE', 'MDAE', 'EVS', 'R2'])
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Polynomial Results\\Main Results.csv', index=False)