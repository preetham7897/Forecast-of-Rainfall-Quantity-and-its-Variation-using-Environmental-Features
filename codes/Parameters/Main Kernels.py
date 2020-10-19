import pandas as pd
from sklearn.model_selection import RepeatedKFold as rkf
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import median_absolute_error as mdae
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import r2_score as r2
import numpy as np
from sklearn.utils import shuffle

def rmse(y_t, y_p):
    return (mse(y_t, y_p))**0.5

data_list = ['All Districts']#, 'Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Ariyalur','Chennai','Coimbatore','Cuddalore','Dharmapuri','Dindigul','Erode','Kancheepuram','Karur','Madurai','Nagapattinam','Namakkal','Perambalur','Pudukkottai','Ramanathapuram','Salem','Sivaganga','Thanjavur','Theni','The Nilgiris','Thiruvallur','Thiruvarur','Thoothukkudi','Tiruchirapalli','Tirunelveli','Tiruvannamalai','Vellore','Viluppuram','Virudhunagar']
ker = ['linear', 'poly', 'rbf', 'sigmoid']
#columns for result
dl = []
kernel = []
mse_ts = []
rmse_ts = []
mae_ts = []
mdae_ts = []
evs_ts = []
r2_ts = []
rkf = rkf(n_splits=10, n_repeats=10)
for i in data_list:
    data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\' + i + '.csv')
    data = shuffle(data)
    #Error measures init
    k = []
    mse_k = []
    rmse_k = []
    mae_k = []
    mdae_k = []
    evs_k = []
    r2_k = []
    c = 0
    #Repeated K Fold Cross Validation
    for tr_i, ts_i in rkf.split(data):
        train, test = data.iloc[tr_i], data.iloc[ts_i]
        train_x = train.drop(columns=['District', 'Index', 'Rainfall', 'Minimum Temperature'])
        train_y = train['Rainfall']
        test_x = test.drop(columns=['District', 'Index', 'Rainfall', 'Minimum Temperature'])
        test_y = test['Rainfall']
        for j in ker:
            print(i, c, j)
            svr = SVR(kernel=j, C=1, epsilon=0.1)
            svr.fit(train_x, train_y)
            svr_p = svr.predict(test_x)
            #Error values
            k.append(j)
            mse_k.append(mse(test_y, svr_p))
            rmse_k.append(rmse(test_y, svr_p))
            mae_k.append(mae(test_y, svr_p))
            mdae_k.append(mdae(test_y, svr_p))
            evs_k.append(evs(test_y, svr_p))
            r2_k.append(r2(test_y, svr_p))
        c += 1
    t = {}
    t['Kernel'] = k
    t['MSE'] = mse_k
    t['RMSE'] = rmse_k
    t['MAE'] = mae_k
    t['MDAE'] = mdae_k
    t['EVS'] = evs_k
    t['R2'] = r2_k
    tf = pd.DataFrame(t, columns=['Kernel', 'MSE', 'RMSE', 'MAE', 'MDAE', 'EVS', 'R2'])
    for j in ker:
        temp = tf[tf['Kernel'] == j]
        dl.append(i)
        kernel.append(j)
        mse_ts.append(np.mean(list(temp['MSE'])))
        rmse_ts.append(np.mean(list(temp['RMSE'])))
        mae_ts.append(np.mean(list(temp['MAE'])))
        mdae_ts.append(np.mean(list(temp['MDAE'])))
        evs_ts.append(np.mean(list(temp['EVS'])))
        r2_ts.append(np.mean(list(temp['R2'])))
d = {}
d['Data'] = dl
d['Kernel'] = kernel
d['MSE'] = mse_ts
d['RMSE'] = rmse_ts
d['MAE'] = mae_ts
d['MDAE'] = mdae_ts
d['EVS'] = evs_ts
d['R2'] = r2_ts
df = pd.DataFrame(d, columns=['Data', 'Kernel', 'MSE', 'RMSE', 'MAE', 'MDAE', 'EVS', 'R2'])
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\SVR Results\\Main Results - GM.csv', index=False)