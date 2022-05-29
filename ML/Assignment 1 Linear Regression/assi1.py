import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv("hours.csv")
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

from sklearn.linear_model import LinearRegression

Regressor = LinearRegression()
Regressor.fit(x,y)
r_sq =Regressor.score(x,y)

print('Coefficient of determination:',r_sq)
print('intercept:',Regressor.intercept_)
print('slope:',Regressor.coef_)
print ("Accuracy:",Regressor.score(x,y)*100)
y_pred=Regressor.predict([[8]])
print ("y_pred:",y_pred)
plt.plot(x,y,'o')
plt.plot(x,Regressor.predict(x))
plt.show()
