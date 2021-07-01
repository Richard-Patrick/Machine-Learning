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
# import the SVD
from sklearn.decomposition import TruncatedSVD
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
SVD=TruncatedSVD(n_components=3)
SVD.fit(scaler) 
x_SVD=SVD.transform(scaler)



# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0])

ax.scatter(x_SVD[:, 0], x_SVD[:, 1], x_SVD[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')


plt.show()