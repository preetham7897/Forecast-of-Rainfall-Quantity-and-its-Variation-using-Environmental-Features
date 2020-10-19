import pandas as pd

data_list = ['Cluster 0', 'Cluster 1', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6']
d = {}
for i in data_list:
    data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\Cluster ' + i + '.csv')
    