# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 13:41:04 2021

@author: alpha
"""

# import pandas for importing csv files
import pandas as pd
# Import process data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
# import the PCA
from sklearn.decomposition import PCA
#data visualization
import matplotlib.pyplot as plt    
from mpl_toolkits.mplot3d import Axes3D



# reading csv files
df = pd.read_csv('../data/IRIS.csv',sep=',')
#process data
labelEncoder = preprocessing.LabelEncoder()
df.Species = labelEncoder.fit_transform(df.Species)
Scaler=StandardScaler()
Scaler.fit(df)
scaler=Scaler.transform(df)

y = df.values[:,-1:].flatten().astype(int)


fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
pca=PCA(n_components=3)
pca.fit(scaler)
x_pca=pca.transform(scaler)

print(y)
print(x_pca)

# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0])

ax.scatter(x_pca[:, 0], x_pca[:, 1], x_pca[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()