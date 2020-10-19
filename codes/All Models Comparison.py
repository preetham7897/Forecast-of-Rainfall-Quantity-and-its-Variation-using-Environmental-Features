import pandas as pd
from sklearn.model_selection import RepeatedKFold as rkf
from sklearn import linear_model as lm
from sklearn.svm import SVR
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

dist = ['Dindigul', 'Erode', 'Karur', 'Thoothukkudi', 'Ariyalur', 'Chennai', 'Cuddalore', 'Kancheepuram', 'Namakkal', 'Perambalur', 'Salem', 'Thiruvallur', 'Viluppuram', 'Coimbatore', 'Madurai', 'Ramanathapuram', 'Theni', 'The Nilgiris', 'Virudhunagar', 'Tirunelveli', 'Nagapattinam', 'Pudukkottai', 'Sivaganga', 'Thanjavur', 'Thiruvarur', 'Tiruchirapalli', 'Dharmapuri', 'Tiruvannamalai', 'Vellore']
gen = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\All Districts.csv')
parameters = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Parameters\\Parameters.csv')
gen_poly = pf(degree=4)
ds_poly = pf(degree=2)
gen_pr = lm.LinearRegression()
clus_pr = lm.LinearRegression()
#MSE for mean
mse_ts_ds = []
mse_ts_clus = []
mse_ts_gen = []
#RMSE for mean
rmse_ts_ds = []
rmse_ts_clus = []
rmse_ts_gen = []
#MAE for mean
mae_ts_ds = []
mae_ts_clus = []
mae_ts_gen = []
#MDAE for mean
mdae_ts_ds = []
mdae_ts_clus = []
mdae_ts_gen = []
#EVS for mean
evs_ts_ds = []
evs_ts_clus = []
evs_ts_gen = []
#R2 for mean
r2_ts_ds = []
r2_ts_clus = []
r2_ts_gen = []
rkf = rkf(n_splits=10, n_repeats=10)
#Iterating through Districts
for i in dist:
    ds = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\' + i + '.csv')
    ds = shuffle(ds)
    gen = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\All Districts.csv')
    gen = shuffle(gen)
    if i in ('Dindigul', 'Erode', 'Karur', 'Thoothukkudi'):
        c = 0
        clus_poly = pf(degree=3)
    elif i in ('Ariyalur', 'Chennai', 'Cuddalore', 'Kancheepuram', 'Namakkal', 'Perambalur', 'Salem', 'Thiruvallur', 'Viluppuram'):
        c = 1
        clus_poly = pf(degree=4)
    elif i in ('Coimbatore', 'Madurai', 'Ramanathapuram', 'Theni', 'The Nilgiris', 'Virudhunagar'):
        c = 2
        clus_poly = pf(degree=3)
    elif i in ('Tirunelveli'):
        c = 3
        clus_poly = pf(degree=2)
    elif i in ('Nagapattinam', 'Pudukkottai', 'Sivaganga', 'Thanjavur', 'Thiruvarur', 'Tiruchirapalli'):
        c = 4
        clus_poly = pf(degree=3)
    else:
        c = 5
        clus_poly = pf(degree=3)
    if i in ('Kancheepuram', 'Tiruvannamalai', 'Vellore'):
        ds_mo = SVR(kernel='rbf', C=1, epsilon=0.1)
    else:
        ds_mo = lm.LinearRegression()
    clus = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\Cluster ' + str(c) + '.csv')
    clus = shuffle(clus)
    gen_ds_n = gen[gen['District'] != i]
    clus_ds_n = clus[clus['District'] != i]
    gen_ds = gen[gen['District'] == i]
    clus_ds = clus[clus['District'] == i]
    #MSE for mean
    mse_ds = []
    mse_clus = []
    mse_gen = []
    #RMSE for mean
    rmse_ds = []
    rmse_clus = []
    rmse_gen = []
    #MAE for mean
    mae_ds = []
    mae_clus = []
    mae_gen = []
    #MDAE for mean
    mdae_ds = []
    mdae_clus = []
    mdae_gen = []
    #EVS for mean
    evs_ds = []
    evs_clus = []
    evs_gen = []
    #R2 for mean
    r2_ds = []
    r2_clus = []
    r2_gen = []
    p = 0
    #Repeated K Fold Cross Validation
    for tr_i, ts_i in rkf.split(ds):
        print(i, c, p)
        p += 1
        train, test = ds.iloc[tr_i], ds.iloc[ts_i]
        l = list(test['Index'])
        train_ds_x = train.drop(columns=['Index', 'District', 'Rainfall'])
        test_ds_x = test.drop(columns=['Index', 'District', 'Rainfall'])
        test_ds_y = test['Rainfall']
        train_ds_y = train['Rainfall']
        clus_tr = clus_ds[~clus_ds['Index'].isin(l)]
        clus_ds_n = clus_ds_n.append(clus_tr)
        clus_ts = clus_ds[clus_ds['Index'].isin(l)]
        gen_tr = gen_ds[~gen_ds['Index'].isin(l)]
        gen_ds_n = gen_ds_n.append(gen_tr)
        gen_ts = gen_ds[gen_ds['Index'].isin(l)]
        print('Gen : ', list(set(list(gen_tr['Index'])).intersection(list(gen_ts['Index']))))
        print('Clus : ', list(set(list(clus_tr['Index'])).intersection(list(clus_ts['Index']))))
        train_clus_x = clus_ds_n.drop(columns=['Index', 'District', 'Rainfall'])
        train_clus_y = clus_ds_n['Rainfall']
        test_clus_x = clus_ts.drop(columns=['Index', 'District', 'Rainfall'])
        test_clus_y = clus_ts['Rainfall']
        train_gen_x = gen_ds_n.drop(columns=['Index', 'District', 'Rainfall'])
        train_gen_y = gen_ds_n['Rainfall']
        test_gen_x = gen_ts.drop(columns=['Index', 'District', 'Rainfall'])
        test_gen_y = gen_ts['Rainfall']
        poly_gen_tr_x = gen_poly.fit_transform(train_gen_x)
        poly_gen_ts_x = gen_poly.fit_transform(test_gen_x)
        if i not in ('Kancheepuram', 'Tiruvannamalai', 'Vellore', 'Dharmapuri', 'Dindigul', 'Madurai', 'Ramanathapuram', 'Theni', 'Tirunelveli', 'Virudhunagar'):
            train_ds_x = ds_poly.fit_transform(train_ds_x)
            test_ds_x = ds_poly.fit_transform(test_ds_x)
        #Fitting the models
        ds_mo.fit(train_ds_x, train_ds_y)
        poly_clus_tr_x = clus_poly.fit_transform(train_clus_x)
        poly_clus_ts_x = clus_poly.fit_transform(test_clus_x)
        clus_pr.fit(poly_clus_tr_x, train_clus_y)
        gen_pr.fit(poly_gen_tr_x, train_gen_y)
        #Predicting test
        ds_p = ds_mo.predict(test_ds_x)
        gen_p = gen_pr.predict(poly_gen_ts_x)
        clus_p = clus_pr.predict(poly_clus_ts_x)
        #MSE for models
        mse_ds.append(mse(test_ds_y, ds_p))
        mse_clus.append(mse(test_clus_y, clus_p))
        mse_gen.append(mse(test_gen_y, gen_p))
        #RMSE for models
        rmse_ds.append(rmse(test_ds_y, ds_p))
        rmse_clus.append(rmse(test_clus_y, clus_p))
        rmse_gen.append(rmse(test_gen_y, gen_p))
        #MAE for models
        mae_ds.append(mae(test_ds_y, ds_p))
        mae_clus.append(mae(test_clus_y, clus_p))
        mae_gen.append(mae(test_gen_y, gen_p))
        #MDAE for models
        mdae_ds.append(mdae(test_ds_y, ds_p))
        mdae_clus.append(mdae(test_clus_y, clus_p))
        mdae_gen.append(mdae(test_gen_y, gen_p))
        #EVS for models
        evs_ds.append(evs(test_ds_y, ds_p))
        evs_clus.append(evs(test_clus_y, clus_p))
        evs_gen.append(evs(test_gen_y, gen_p))
        #R2 for models
        r2_ds.append(r2(test_ds_y, ds_p))
        r2_clus.append(r2(test_clus_y, clus_p))
        r2_gen.append(r2(test_gen_y, gen_p))
    #Mean of MSE for models
    mse_ts_ds.append(np.mean(mse_ds))
    mse_ts_clus.append(np.mean(mse_clus))
    mse_ts_gen.append(np.mean(mse_gen))
    #Mean of RMSE for models
    rmse_ts_ds.append(np.mean(rmse_ds))
    rmse_ts_clus.append(np.mean(rmse_clus))
    rmse_ts_gen.append(np.mean(rmse_gen))
    #Mean of MAE for models
    mae_ts_ds.append(np.mean(mae_ds))
    mae_ts_clus.append(np.mean(mae_clus))
    mae_ts_gen.append(np.mean(mae_gen))
    #Mean of MDAE for models
    mdae_ts_ds.append(np.mean(mdae_ds))
    mdae_ts_clus.append(np.mean(mdae_clus))
    mdae_ts_gen.append(np.mean(mdae_gen))
    #Mean of EVS for models
    evs_ts_ds.append(np.mean(evs_ds))
    evs_ts_clus.append(np.mean(evs_clus))
    evs_ts_gen.append(np.mean(evs_gen))
    #Mean of R2 for models
    r2_ts_ds.append(np.mean(r2_ds))
    r2_ts_clus.append(np.mean(r2_clus))
    r2_ts_gen.append(np.mean(r2_gen))
#Converting to dict
d = {}
d['District'] = dist
d['MSE - District'] = mse_ts_ds
d['MSE - Cluster'] = mse_ts_clus
d['MSE - Generic'] = mse_ts_gen
d['RMSE - District'] = rmse_ts_ds
d['RMSE - Cluster'] = rmse_ts_clus
d['RMSE - Generic'] = rmse_ts_gen
d['MAE - District'] = mae_ts_ds
d['MAE - Cluster'] = mae_ts_clus
d['MAE - Generic'] = mae_ts_gen
d['MDAE - District'] = mdae_ts_ds
d['MDAE - Cluster'] = mdae_ts_clus
d['MDAE - Generic'] = mdae_ts_gen
d['EVS - District'] = evs_ts_ds
d['EVS - Cluster'] = evs_ts_clus
d['EVS - Generic'] = evs_ts_gen
d['R2 - District'] = r2_ts_ds
d['R2 - Cluster'] = r2_ts_clus
d['R2 - Generic'] = r2_ts_gen
df = pd.DataFrame(d, columns=['District', 'MSE - District', 'MSE - Cluster', 'MSE - Generic', 'RMSE - District', 'RMSE - Cluster', 'RMSE - Generic', 'MAE - District', 'MAE - Cluster', 'MAE - Generic', 'MDAE - District', 'MDAE - Cluster', 'MDAE - Generic', 'EVS - District', 'EVS - Cluster', 'EVS - Generic', 'R2 - District', 'R2 - Cluster', 'R2 - Generic'])
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Model Comparison.csv', index=False)