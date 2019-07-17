import pandas as pd

def mean(y):
    return sum(y)/len(y)

dist = ['Ariyalur','Chennai','Coimbatore','Cuddalore','Dharmapuri','Dindigul','Erode','Kancheepuram','Karur','Madurai','Nagapattinam','Namakkal','Perambalur','Pudukkottai','Ramanathapuram','Salem','Sivaganga','Thanjavur','Theni','The Nilgiris','Thiruvallur','Thiruvarur','Thoothukkudi','Tiruchirapalli','Tirunelveli','Tiruvannamalai','Vellore','Viluppuram','Virudhunagar']
feat = ['Average Temperature', 'Cloud Cover', 'Crop Evapotranspiration', 'Maximum Temperature', 'Minimum Temperature', 'Potential Evapotranspiration', 'Vapour Pressure', 'Wet Day Frequency', 'Rainfall']
for i in dist:
    d = {}
    for j in feat:
        data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Combined Data\\' + i + '.csv')
        m = list(data[j])
        l = []
        for k in m:
            l.append((k-min(m))/(max(m)-min(m)))
        d[j] = l
    d['District'] = list(data['District'])
    d['Index'] = list(range(0,1224))
    df = pd.DataFrame(d, columns=['District', 'Index', 'Average Temperature', 'Cloud Cover', 'Crop Evapotranspiration', 'Maximum Temperature', 'Minimum Temperature', 'Potential Evapotranspiration', 'Vapour Pressure', 'Wet Day Frequency', 'Rainfall'])
    df.to_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Forecast of Rainfall Quantity and its variation using Envrionmental Features\\Data\\Normalized & Combined Data\\' + i + '.csv', index=False)
