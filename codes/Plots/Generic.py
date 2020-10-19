import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn import linear_model as lm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

font = {'family' : 'Times New Roman',
        'size'   : 28}

figure(num=None, figsize=(20, 10))
dist = ['Dindigul', 'Erode', 'Karur', 'Thoothukkudi', 'Ariyalur', 'Chennai', 'Cuddalore', 'Kancheepuram', 'Namakkal', 'Perambalur', 'Salem', 'Thiruvallur', 'Viluppuram', 'Coimbatore', 'Madurai', 'Ramanathapuram', 'Theni', 'The Nilgiris', 'Virudhunagar', 'Tirunelveli', 'Nagapattinam', 'Pudukkottai', 'Sivaganga', 'Thanjavur', 'Thiruvarur', 'Tiruchirapalli', 'Dharmapuri', 'Tiruvannamalai', 'Vellore']
gen = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\All Districts.csv')
test = pd.DataFrame()
for i in dist:
    ds = gen[gen['District'] == i]
    gen = gen[gen['District'] != i]
    ds = shuffle(ds)
    t = ds.head()
    index = list(t['Index'])
    test = test.append(t)
    ds = ds[~ds['Index'].isin(index)]
    gen = gen.append(ds)
pr = lm.LinearRegression()
poly = pf(degree=4)
gen_x = gen.drop(columns=['Index', 'District', 'Rainfall'])
gen_y = gen['Rainfall']
test_x = test.drop(columns=['Index', 'District', 'Rainfall'])
test_y = test['Rainfall']
poly_tr = poly.fit_transform(gen_x)
poly_ts = poly.fit_transform(test_x)
pr.fit(poly_tr, gen_y)
pr_p = pr.predict(poly_ts)
plt.plot(list(range(len(pr_p))), list(test_y), 'b', label='True', linewidth=3)
plt.plot(list(range(len(pr_p))), list(pr_p), 'r', label='Pred', linewidth=3)
plt.xlabel('Data Points - 5 from each District')
plt.ylabel('Rainfall')
plt.legend(loc='upper right')
plt.show()