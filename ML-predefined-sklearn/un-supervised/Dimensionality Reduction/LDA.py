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
# import the LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

X=scaler[:,:-1]
y = df.values[:,-1:].flatten().astype(int)


fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()
LDA=LinearDiscriminantAnalysis()
LDA.fit(X,y) 

x_LDA=LDA.transform(X)


ax.scatter(x_LDA[:, 0],x_LDA[:, 1], c=y, cmap=plt.cm.nipy_spectral,edgecolor='k')


plt.show()
