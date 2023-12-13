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
print("Passengers Lost")
print(df1_lost.groupby('Cabin').size())

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
img = cv2.imread('titanic.jpeg')
imS = cv2.resize(img, (960, 540))                # Resize image
cv2.imshow("output", imS)                       # Show image
cv2.waitKey(0)    

# make a copy of the original image
imageRectangle = imS.copy()
# define the starting and end points of the rectangle
start_point =(300,115)
end_point =(275,448)
# draw the rectangle
cv2.rectangle(imageRectangle, start_point, end_point, (0, 0, 255), thickness= 3, lineType=cv2.LINE_8) 
# display the output
cv2.imshow('imageRectangle', imageRectangle)
cv2.waitKey(0)

