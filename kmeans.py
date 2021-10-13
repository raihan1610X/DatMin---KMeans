import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

X = np.array([
     [2,5],
     [2,5.5],
     [5,3.5],
     [6.5,2.2],
     [7,3.3],
     [3.5,4.8],
     [4,4.5],])

plt.scatter(X[:,0],X[:,1], label='True Position')

kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

print(kmeans.clustercenters)
print(kmeans.labels)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels, cmap='rainbow')