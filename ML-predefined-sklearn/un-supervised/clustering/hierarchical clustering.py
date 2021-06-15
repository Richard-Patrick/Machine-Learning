# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 11:42:45 2021

@author: alpha
"""

# import pandas for importing csv files
import pandas as pd
# import split data
from sklearn.model_selection import train_test_split
# import the Cluster
from sklearn.cluster import AgglomerativeClustering  

import matplotlib.pyplot as plt    


# reading csv files
df = pd.read_csv('../data/Mall_Customers.csv',sep=',')

x = df.iloc[:, [3, 4]].values


cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
Y=cluster.fit_predict(x)

plt.figure(figsize=(10, 7))
plt.scatter(x[:,0], x[:,1], c=cluster.labels_, cmap='rainbow')