# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 12:24:52 2023

@author: pc
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn import metrics

df=pd.read_csv('Crop_recommendation.csv')

df.head()

df['label'].unique()

df['label'].value_counts()


import seaborn as sns
sns.heatmap(df.corr(),annot=True)

features=df[['N','P','K','temperature','humidity','ph','rainfall']]
target=df['label']
labels=df['label']


from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)

from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(Xtrain,Ytrain)

predicted_values = DecisionTree.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
print("DecisionTrees's Accuracy is: ", x*100)   


from sklearn.naive_bayes import GaussianNB

NaiveBayes = GaussianNB()

NaiveBayes.fit(Xtrain,Ytrain)

predicted_values = NaiveBayes.predict(Xtest)
x = metrics.accuracy_score(Ytest, predicted_values)
print("Naive Bayes's Accuracy is: ", x)

print(classification_report(Ytest,predicted_values))

from sklearn.model_selection import cross_val_score

# Cross validation score for NaiveBayes
score = cross_val_score(NaiveBayes,features,target,cv=5)
score

import pickle
# Dump the trained Naive Bayes classifier with Pickle
#NB_pkl_filename = 'NBClassifier.pkl'
# Open the file to save as pkl file
#NB_Model_pkl = open(NB_pkl_filename, 'wb')
#pickle.dump(NaiveBayes, NB_Model_pkl)
# Close the pickle instances
#NB_Model_pkl.close()

import joblib
joblib.dump(NaiveBayes,'NB_Model.obj')

#data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
#prediction = NaiveBayes.predict(data)
#print(prediction)

#data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
#prediction =NaiveBayes.predict(data)
#print(prediction)
