import pandas as pd

clusters = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Clustering\\6 Clusters.csv')
feat = ['Average Temperature', 'Cloud Cover', 'Crop Evapotranspiration', 'Maximum Temperature', 'Minimum Temperature', 'Potential Evapotranspiration', 'Vapour Pressure', 'Wet Day Frequency', 'Rainfall']
columns = ['District', 'Index'] + feat

for i in range(6):
    c = []
    cluster = clusters[clusters['Cluster'] == i]
    dist = list(cluster['District'].unique())
    for j in dist:
        print(i, j)
        dis = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Combined Data\\' + j + '.csv')
        c.append(dis)
    data = pd.concat(c)
    d = {}
    for j in feat:
        l = list(data[j])
        m = []
        for k in l:
            m.append((k - min(l))/(max(l) - min(l)))
        d[j] = m
    d['District'] = list(data['District'])
    d['Index'] = list(data['Index'])
    df = pd.DataFrame(d, columns=columns)
    path = 'C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\Cluster ' + str(i) + '.csv'
    df.to_csv(path, index=False)    