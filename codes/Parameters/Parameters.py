import pandas as pd

def count_k(l):
    k = ['linear', 'poly', 'rbf', 'sigmoid']
    c = []
    for i in k:
        c.append(l.count(i))
    max_c = max(c)
    return k[c.index(max_c)]

def count_m(l):
    m = [2, 3, 4, 5, 6, 7]
    c = []
    for i in m:
        c.append(l.count(i))
    max_c = max(c)
    return m[c.index(max_c)]

def count_d(l):
    d = [2, 3, 4, 5]
    c = []
    for i in d:
        c.append(l.count(i))
    max_c = max(c)
    return d[c.index(max_c)]        

svr = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Parameters\\Main Results - SVR.csv')
dtr = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Parameters\\Main Results - DTR.csv')
pr = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Parameters\\Main Results - Poly.csv')
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
degree = [2, 3, 4, 5]
depth = [2, 3, 4, 5, 6, 7]
data_list = ['All Districts', 'Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Ariyalur','Chennai','Coimbatore','Cuddalore','Dharmapuri','Dindigul','Erode','Kancheepuram','Karur','Madurai','Nagapattinam','Namakkal','Perambalur','Pudukkottai','Ramanathapuram','Salem','Sivaganga','Thanjavur','Theni','The Nilgiris','Thiruvallur','Thiruvarur','Thoothukkudi','Tiruchirapalli','Tirunelveli','Tiruvannamalai','Vellore','Viluppuram','Virudhunagar']
nm = ['MSE', 'RMSE', 'MAE', 'MDAE']
pm = ['EVS', 'R2']
fk = []
fm = []
fd = []
for i in data_list:
    svr_t = svr[svr['Data'] == i]
    dtr_t = dtr[dtr['Data'] == i]
    pr_t = pr[pr['Data'] == i]
    svr_l = []
    dtr_l = []
    pr_l = []
    for j in nm:
        svr_c = list(svr_t[j])
        dtr_c = list(dtr_t[j])
        pr_c = list(pr_t[j])
        svr_mi = min(svr_c)
        dtr_mi = min(dtr_c)
        pr_mi = min(pr_c)
        svr_l.append(kernel[svr_c.index(svr_mi)])
        dtr_l.append(depth[dtr_c.index(dtr_mi)])
        pr_l.append(degree[pr_c.index(pr_mi)])
    for j in pm:
        svr_c = list(svr_t[j])
        dtr_c = list(dtr_t[j])
        pr_c = list(pr_t[j])
        svr_ma = max(svr_c)
        dtr_ma = max(dtr_c)
        pr_ma = max(pr_c)
        svr_l.append(kernel[svr_c.index(svr_ma)])
        dtr_l.append(depth[dtr_c.index(dtr_ma)])
        pr_l.append(degree[pr_c.index(pr_ma)])
    fk.append(count_k(svr_l))
    fm.append(count_m(dtr_l))
    fd.append(count_d(pr_l))
d = {}
d['Data'] = data_list
d['SVR Kernel'] = fk
d['DTR Max Depth'] = fm
d['PR Degree'] = fd
df = pd.DataFrame(d, columns=['Data', 'SVR Kernel', 'DTR Max Depth', 'PR Degree'])
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Parameters\\Parameters.csv', index=False)       