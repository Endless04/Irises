import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv('Iris.csv')
x=dataset.iloc[:,1:5].values
y=dataset.iloc[:,5]
from sklearn.preprocessing import LabelEncoder
z=LabelEncoder()
y=z.fit_transform(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
from sklearn.tree import DecisionTreeRegressor
regressor1=DecisionTreeRegressor(random_state=0)
regressor1.fit(x_train,y_train)
y_pred1=regressor1.predict(x_test)