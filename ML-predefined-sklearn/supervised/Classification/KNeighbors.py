# -*- coding: utf-8 -*-
"""
Created on Mon May 17 21:34:38 2021

@author: Richard
"""

# import pandas for importing csv files
import pandas as pd
# import split data
from sklearn.model_selection import train_test_split
# import the Classifier
from sklearn import neighbors
# import for error check
from sklearn import metrics 



# reading csv files
df = pd.read_csv('../../data/IRIS.csv',sep=',')

# split X Y
X = df.values[:, 1:4]
Y = df.values[:,4]

# split Train and Test
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

# create a Classifier object
Classifier = neighbors.KNeighborsClassifier()

# fit the Classifier with X Train and Y Train
Classifier.fit(X_train,y_train)

# predict the Y value with X Test
y_pred = Classifier.predict(X_test)

# The accuracy of the classification model
print(metrics.accuracy_score(y_test,y_pred))

