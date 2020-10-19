import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\All Districts.csv')
dist = ['Ariyalur','Chennai','Coimbatore','Cuddalore','Dharmapuri','Dindigul','Erode','Kancheepuram','Karur','Madurai','Nagapattinam','Namakkal','Perambalur','Pudukkottai','Ramanathapuram','Salem','Sivaganga','Thanjavur','Theni','The Nilgiris','Thiruvallur','Thiruvarur','Thoothukkudi','Tiruchirapalli','Tirunelveli','Tiruvannamalai','Vellore','Viluppuram','Virudhunagar']
feat = ['Average Temperature', 'Cloud Cover', 'Crop Evapotranspiration', 'Maximum Temperature', 'Minimum Temperature', 'Potential Evapotranspiration', 'Vapour Pressure', 'Wet Day Frequency', 'Rainfall']
d_med = {}
for i in feat:
    f_med = []
    dat = data[['District', i]]
    for j in dist:
        print(i, j)
        td = dat[dat['District'] == j]
        t = list(td[i])
        f_med.append(np.median(t))
    d_med[i] = f_med
d_med['District'] = dist
columns = ['District'] + feat
df_med = pd.DataFrame(d_med, columns=columns)
km_med = KMeans(n_clusters=6)
km_med.fit(df_med.drop(columns=['District']))
l_med = list(km_med.labels_)
df_med['Cluster'] = l_med
df_med.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Results\\Clustering\\6 Clusters.csv', index=False)