#import packages
import pandas as pd
import numpy as np

# Read dataset
dataset=pd.read_csv("tree1.csv")
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,5]

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
x=x.apply(le.fit_transform)
print(x)
#1100

#import Decesion Tree Classifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# Create decision tree classifer object
regressor=DecisionTreeClassifier()
# Train model
regressor.fit(x.iloc[:,1:5].values,y)
x_in=np.array([1,1,0,0])
y_pred=regressor.predict([x_in])
print(y_pred)
from sklearn import tree
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,12))
tree.plot_tree(regressor, filled = True, rounded = True)
plt.show()













