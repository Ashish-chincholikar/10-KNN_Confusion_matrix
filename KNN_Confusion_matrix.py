# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:26:35 2024

@author: Ashish
KNN_Confusion_Matrix
"""

import pandas as pd
import numpy as np

wbcd = pd.read_csv("C:/Supervised_ML/KNN_Confusion_matrix/wbcd.csv")
#There are 569rows and 32 columns
wbcd.describe()
#in output column there is only B for Benien and M for Malignant
#Let us first convert it as Benien and Malignant
wbcd['diagnosis'] = np.where(wbcd['diagnosis']=='B', 'Beniegn' , wbcd['diagnosis'])

#in wbcd there is column named 'diagnosis' , where ever there is 'B' 
#replaced by 'Beniegn' 
#similarly where ever there is M in the Same Column replace with
#"Malignant"
wbcd['diagnosis'] = np.where(wbcd['diagnosis']=='M', 'Malignant' , wbcd['diagnosis'])
 
#0th column is patient ID let us drop it
wbcd = wbcd.iloc[: , 1:32]

#Normalization
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

#Now let us apply this function to the dataframe
wbcd_n = norm_func(wbcd.iloc[: , 1:32])
#Because now 0th column is output or label it is not considered hence 1:32

#let us now apply X as input and y as output
X = np.array(wbcd_n.iloc[:,:])
#Since in wbcd_n we are already excluding output column , hence all rows and columns
y = np.array(wbcd['diagnosis'])

#Now let us split the data into training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#here you are passing X, y instead Dataframe handle
#there could chance of unbalancing of data.
#let us assume you have 100 data points, out of which 80 NC and 20 Cancer
#These Data points must be equally distributed
#there is statified sampling concept is used
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
pred

#Now let us evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(pred, y_test))
pd.crosstab(pred, y_test)

#Now let us check the aplicability of the model 
#i.e miss classification , Actual patient is malignant
#i.e cancer patient but predicted in Benien is 1 
#Actual patient is Benien and predicted as cancer patient is 5
#hence this model is not acceptable

#let us try to select the correct value of k
acc = []
#Running KNN algorithm for k=3 to 50 in the step of 2
#k value selected is odd value
for i in range(3,50,2):
    #Declare the model 
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    train_acc = np.mean(neigh.predict(X_train)==y_train)
    test_acc = np.mean(neigh.predict(X_test)==y_test)
    acc.append([train_acc,test_acc])

#If you will see the acc . it has got two accuracy , i[0]-train-acc
#i[1] = test_acc
# to plot the graph of train_acc and test_acc
import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2) , [i[0] for i in acc] , "ro-")
plt.plot(np.arange(3,50,2) , [i[1] for i in acc] , "bo-")
#There are 3,5,7 and 9 are possible values where accuracy is good
#let us check for k=3
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
accuracy_score(pred, y_test)
pd.crosstab(pred , y_test)
#ie.miss classification , actual  pateint is malignant
#i.e. cancer patient but predicted is Benien is 1
#Actual patient is Benien and predicted as cancer patient is 2
#henece this model is not acceptale
#for 5 same sinario
# for k=7 we are getting zero false positive and good accuracy
#hence k=7 is appropriate value of k
