#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 23:50:13 2023

@author: pulsaragunawardhana
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
from sklearn import preprocessing
import cv2

df = pd.read_csv('tested.csv')

df.drop(['Name','PassengerId','Ticket'], axis = 1, inplace = True)

print(df.info())
print(df.describe())

mean_age = np.mean(df.Age)
mean_fare = np.mean(df.Fare)

print('The mean age of the Titanic Passengers :', mean_age)

df['Cabin'] = df['Cabin'].str[0].map({'B': 'Upper Cabin', 'A': 'Highest Cabin', 'D': 'Lower Cabin','F':'Lowest Cabin'})
values = {"Age": mean_age, "Cabin": "Lower Cabin",'Fare': mean_fare}
df.fillna(value=values, inplace = True)

df1_survived = df[df.Survived==1]
df1_lost = df[df.Survived==0]
print("Passengers Survived")
print(df1_survived.groupby('Cabin').size())
cv_survived = df1_survived.groupby('Cabin').size().reset_index()
print("Passengers Lost")
print(df1_lost.groupby('Cabin').size())
cv_lost = df1_lost.groupby('Cabin').size().reset_index()

print(df.info())
print("Cabins")
print(df.Cabin.unique())
le = preprocessing.LabelEncoder() 
df['Cabin']= le.fit_transform(df['Cabin']) 
df['Embarked']= le.fit_transform(df['Embarked']) 
df['Sex']= le.fit_transform(df['Sex']) 

df.groupby('Pclass')['Fare'].plot(legend=True)
df_survived = df[df.Survived==1]
df_lost = df[df.Survived==0]

plt.figure()
df_survived['Pclass'].value_counts().plot(kind='bar')
plt.title('The Class Distribution of the Survivors')
plt.figure()
sns.heatmap(df.corr(), vmin=-1, vmax=1,annot=True,cmap="rocket_r")
plt.show()

# Read Images
cv2.namedWindow("output", cv2.WINDOW_NORMAL)
img = cv2.imread('titanic_edited.png')
imS = cv2.resize(img, (960, 540))                # Resize image   

# make a copy of the original image
imageLine = imS.copy()

#The lowest cabin level display 
text = 'Passengers Lost:'  + str(cv_lost.iloc[1][0]) 
text_2 = 'Passengers Survived:' + str(cv_survived.iloc[1][0])
pointA = (254,468)
pointB = (400,468)
org = (401,475)
org_2 = (401,488)
cv2.arrowedLine(imageLine, pointB, pointA, (169, 169, 169), thickness=2)
cv2.putText(imageLine, text, org, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,0,0),thickness=1)
cv2.putText(imageLine, text_2, org_2, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (0,0,0),thickness=1)


#The middle cabin level display 
text = 'Passengers Lost:'  + str(cv_lost.iloc[2][0]) 
text_2 = 'Passengers Survived:' + str(cv_survived.iloc[2][0])
pointB = (400,408)
pointA = (257,408)
org = (401,413)
org_2 = (401,426)
cv2.arrowedLine(imageLine, pointB, pointA, (169, 169, 169), thickness=2)
cv2.putText(imageLine, text, org, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255,255,255),thickness=1)
cv2.putText(imageLine, text_2, org_2, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255,255,255),thickness=1)


#The upper cabin level display 
text = 'Passengers Lost:'  + str(cv_lost.iloc[0][0]) 
text_2 = 'Passengers Survived:' + str(cv_survived.iloc[0][0])
pointB = (486,336)
pointA = (378,336)
org = (486,341)
org_2 = (486,354)
cv2.arrowedLine(imageLine, pointB, pointA, (169, 169, 169), thickness=2)
cv2.putText(imageLine, text, org, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255,255,255),thickness=1)
cv2.putText(imageLine, text_2, org_2, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.5, color = (255,255,255),thickness=1)


cv2.imshow('Titanic EDA Analysis', imageLine)
cv2.waitKey(0)
cv2.destroyAllWindows()



