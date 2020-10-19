import pandas as pd

clust = ['Cluster 0','Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5']
d = {}
month = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
m = []
y = []
for i in range(1901, 2003):
    m.extend(month)
    for j in month:
        y.append(i)
print(len(y))
print(len(m))
for i in clust:
    print(i)
    data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\' + i + '.csv')
    d = list(data['District'].unique())
    mon = []
    year = []
    for j in d:
        mon.extend(m)
        year.extend(y)
    data['Month'] = mon
    data['Years'] = year
    data.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\' + i + '.csv', index=False)