import pandas as pd

def count_m(l):
    m = ['Multiple Linear Regression', 'Support Vector Regression', 'Decision Tree Regression', 'Polynomial Regression']
    c = []
    for i in m:
        c.append(l.count(i))
    max_c = max(c)
    return m[c.index(max_c)]       

result = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Best Model per Data\\Main Results.csv')
parameters = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Parameters\\Parameters.csv')
method = ['Multiple Linear Regression', 'Support Vector Regression', 'Decision Tree Regression', 'Polynomial Regression']
data_list = ['All Districts', 'Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Ariyalur','Chennai','Coimbatore','Cuddalore','Dharmapuri','Dindigul','Erode','Kancheepuram','Karur','Madurai','Nagapattinam','Namakkal','Perambalur','Pudukkottai','Ramanathapuram','Salem','Sivaganga','Thanjavur','Theni','The Nilgiris','Thiruvallur','Thiruvarur','Thoothukkudi','Tiruchirapalli','Tirunelveli','Tiruvannamalai','Vellore','Viluppuram','Virudhunagar']
nm = ['MSE', 'RMSE', 'MAE', 'MDAE']
pm = ['EVS', 'R2']
fm = []
para = []
for i in data_list:
    res_t = result[result['Data'] == i]
    param = parameters[parameters['Data'] == i]
    res_l = []
    for j in nm:
        res_c = list(res_t[j])
        res_mi = min(res_c)
        res_l.append(method[res_c.index(res_mi)])
    for j in pm:
        res_c = list(res_t[j])
        res_ma = max(res_c)
        res_l.append(method[res_c.index(res_ma)])
    fm.append(count_m(res_l))
    if count_m(res_l) == method[0]:
        para.append('-')
    elif count_m(res_l) == method[1]:
        para.append(list(param['SVR Kernel'])[0])
    elif count_m(res_l) == method[2]:
        para.append(list(param['DTR Max Depth'])[0])
    else:
        para.append(list(param['PR Degree'])[0])
d = {}
d['Data'] = data_list
d['Method'] = fm
d['Parameter'] = para
df = pd.DataFrame(d, columns=['Data', 'Method', 'Parameter'])
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Best Model per Data\\Min Results.csv', index=False)       