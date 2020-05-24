#!/usr/bin/env python
# coding: utf-8




import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import joblib


dataset = pd.read_csv('/Code/USA_Housing.csv')




X = dataset[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = dataset['Price']



dcorr = dataset.corr()



X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 42)



model = LinearRegression()

model.fit(X_train , y_train)



joblib.dump(model ,'/Model/USA_Housing_model.pk1')




