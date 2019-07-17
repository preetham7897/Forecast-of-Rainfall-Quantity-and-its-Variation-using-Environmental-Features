import pandas as pd

def mean(y):
    return sum(y)/len(y)

feat = ['Average Temperature', 'Cloud Cover', 'Crop Evapotranspiration', 'Maximum Temperature', 'Minimum Temperature', 'Potential Evapotranspiration', 'Vapour Pressure', 'Wet Day Frequency', 'Rainfall']
data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\All District Combined.csv')
d = {}
for i in feat:
    m = list(data[i])
    l = []
    o = 0
    for j in m:
        o += 1
        print(i, o)
        l.append((j-min(m))/(max(m)-min(m)))
    d[i] = l
d['District'] = list(data['District'])
l = list(range(0, 1224))
m = []
for i in range(29):
    m.append(l)
n = []
for i in m:
    n.extend(i)
d['Index'] = n
df = pd.DataFrame(d, columns=['District', 'Index', 'Average Temperature', 'Cloud Cover', 'Crop Evapotranspiration', 'Maximum Temperature', 'Minimum Temperature', 'Potential Evapotranspiration', 'Vapour Pressure', 'Wet Day Frequency', 'Rainfall'])
df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\\Data\\Normalized & Combined Data\\All Districts.csv', index=False)