# -*- coding: utf-8 -*-
"""
Created on Fri May 21 18:07:10 2021

@author: Richard
"""

# import pandas for importing csv files
import pandas as pd
# import split data
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
# import the Regressor
from sklearn.linear_model import ElasticNet
# import for error check
from sklearn import metrics 



# reading csv files
df = pd.read_csv('../../data/IRIS.csv',sep=',')

# changing column name 
df=df.rename(columns = {"class":"Species"})


#mapping name for float value
mapping = {
    'Iris-setosa' : 1,
    'Iris-versicolor' : 2,
    'Iris-virginica' : 3
}



# split X Y
X = df.values[:, 1:4]
Y = df.Species.replace(mapping).values


# split Train and Test
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)

# create a Regressor object
Regressor = ElasticNet()

# Training Regressor
Regressor.fit(X_train,y_train)

# predict the Y value with X Test
y_pred = Regressor.predict(X_test)





#Regression Accuracy with test set
print(metrics.r2_score(y_test, y_pred))

#Accuracy using cross validation (KFold method)
y_pred_kf_lr = cross_val_predict(Regressor, X, Y, cv=10 )

#Accuracy with cross validation (KFold method)
print(metrics.r2_score(Y, y_pred_kf_lr))

