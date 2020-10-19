import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from matplotlib.pyplot import figure

font = {'family' : 'Times New Roman',
        'size'   : 28}

plt.rc('font', **font)
figure(num=None, figsize=(20, 10))
clust = ['Cluster 0','Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5']
mon = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
d = {}
d['Month'] = mon
for i in clust:
    data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\' + i + '.csv')
    r = []
    for j in mon:
        temp = data[data['Month'] == j]
        rain = list(temp['Rainfall'])
        r.append(np.median(rain))
    d[i] = r
columns = ['Year'] + clust
df = pd.DataFrame(d, columns=columns)
x = list(range(0, 12))
y0 = list(df['Cluster 0'])
y1 = list(df['Cluster 1'])
y2 = list(df['Cluster 2'])
y3 = list(df['Cluster 3'])
y4 = list(df['Cluster 4'])
y5 = list(df['Cluster 5'])
f0 = interp1d(x, y0, kind='quadratic')
f1 = interp1d(x, y1, kind='quadratic')
f2 = interp1d(x, y2, kind='quadratic')
f3 = interp1d(x, y3, kind='quadratic')
f4 = interp1d(x, y4, kind='quadratic')
f5 = interp1d(x, y5, kind='quadratic')
x_new = np.linspace(min(x), max(x), 500)
y0_smooth = f0(x_new)
y1_smooth = f1(x_new)
y2_smooth = f2(x_new)
y3_smooth = f3(x_new)
y4_smooth = f4(x_new)
y5_smooth = f5(x_new)
y0_mean = np.mean(y0_smooth)
y1_mean = np.mean(y1_smooth)
y2_mean = np.mean(y2_smooth)
y3_mean = np.mean(y3_smooth)
y4_mean = np.mean(y4_smooth)
y5_mean = np.mean(y5_smooth)
y0_m = []
y1_m = []
y2_m = []
y3_m = []
y4_m = []
y5_m = []
for i in range(0, len(y0_smooth)):
    y0_m.append(y0_mean)
for i in range(0, len(y1_smooth)):
    y1_m.append(y1_mean)
for i in range(0, len(y2_smooth)):
    y2_m.append(y2_mean)
for i in range(0, len(y3_smooth)):
    y3_m.append(y3_mean)
for i in range(0, len(y4_smooth)):
    y4_m.append(y4_mean)
for i in range(0, len(y5_smooth)):
    y5_m.append(y5_mean)
plt.plot(x_new, y0_smooth, color='red', label='Cluster 1', alpha=0.4, linewidth=3)
#plt.plot(x_new, y1_smooth, color='orange', label='Cluster 2', alpha=0.4, linewidth=3)
plt.plot(x_new, y2_smooth, color='green', label='Cluster 3', alpha=0.4, linewidth=3)
plt.plot(x_new, y3_smooth, color='blue', label='Cluster 4', alpha=0.4, linewidth=3)
#plt.plot(x_new, y4_smooth, color='magenta', label='Cluster 5', alpha=0.4, linewidth=3)
#plt.plot(x_new, y5_smooth, color='black', label='Cluster 6', alpha=0.4, linewidth=3)
plt.plot(x_new, y0_m, '--', color='red', linewidth=3)
#plt.plot(x_new, y1_m, '--', color='orange', linewidth=3)
plt.plot(x_new, y2_m, '--', color='green', linewidth=3)
plt.plot(x_new, y3_m, '--', color='blue', linewidth=3)
#plt.plot(x_new, y4_m, '--', color='magenta', linewidth=3)
#plt.plot(x_new, y5_m, '--', color='black', linewidth=3)
plt.xlabel('Months')
plt.ylabel('Normalized Rainfall')
plt.legend(loc='upper left')
plt.xticks(range(0, 12), mon)
plt.grid(color='black', linestyle='-.', linewidth=2, alpha=0.3)
plt.show()