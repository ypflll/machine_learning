#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on Sun Apr 9 2017
Logistic Regression - Student Enrolling Admit
@author: pavle
"""

from sklearn import linear_model
import numpy as np

fr = open("./binary.csv",'r')

label = []
data = []

#skip the first line
all_lines = fr.readlines()
all_lines = all_lines[1:]

#import data and label
for line in all_lines:
	line_array = line.strip().split(',')
	label.append(int(line_array[0]))
	data.append([float(1), float(line_array[1]), float(line_array[2]), float(line_array[3])])

m,n = np.shape(data)

#standardization
data_mat = np.mat(data)
for i in range(n):
	if 0 != i:
		data_mat[:,i] = (data_mat[:,i] - data_mat[:,i].mean())/data_mat[:,i].std()

#90% to train data and 10% to test
train_data = data_mat[0:m*9/10]
train_label = label[0:m*9/10]
test_data = data_mat[m*9/10:]
test_label = label[m*9/10:]

#use the logistic regression model to fit and predict
lr = linear_model.LogisticRegression()
lr.fit(train_data, train_label)
predicts = lr.predict(test_data)

#calculate the accuracy
accuracy = 0
for (predict,label) in zip(predicts,test_label):
	if predict == label:
		accuracy = accuracy + 1
accuracy = float(accuracy * 100/40)

print predict, accuracy
