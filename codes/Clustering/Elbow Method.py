import pandas as pd
from sklearn.cluster import KMeans
import pylab

data = pd.read_csv('C:\\Users\\Preetham G\\Documents\\Research Projects\\Ensemble Project\\Data\\Converted Data\\Main Data - House 1.csv')
sum_of_sq = []
k = list(range(2,29))
for i in k:
    print(i)
    km = KMeans(n_clusters=i)
    km = km.fit(data)
    sum_of_sq.append(km.inertia_)
pylab.plot(k, sum_of_sq, 'bx-')
pylab.xlabel('Centers')
pylab.ylabel('Sum of Squared Distances')
pylab.plot()