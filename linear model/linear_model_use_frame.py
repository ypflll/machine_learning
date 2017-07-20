#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Created on  
@author: pavle
@e-mail: pavle_yao@yahoo.com

This script implement linear model using sklearn and tensot flow.
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import tensorflow as tf

def load_data(path):
    data = []
    label = []

    f_in = open(path + 'ex1x.txt')
    for line in f_in.readlines():
        x = line.strip().split()
        data.append([float(x[0]), float(x[1])])

    f_in = open(path + 'ex1y.txt')
    for line in f_in.readlines():
        y = line.strip().split()
        label.append(float(y[0]))

    return data, label

def standardization(mat,col):
    mat[:,col]=(mat[:,col]-mat[:,col].mean())/mat[:,col].std()
    return mat

def r_square(x_mat, y_mat, weights, bias):
    y_hat= x_mat * np.mat(weights) + bias
    sse = float(np.sum(pow((y_mat-y_hat).getA(), 2)))
    sst = float(y_mat.var()*len(x_mat))  
    return 1 - sse / sst

def cost(x_mat, y_mat, weights, bias):
    error = x_mat * np.mat(weights) + bias - y_mat
    cost = float(np.sum(pow(error.getA(), 2), axis = 0)) / (2*len(x_mat))
    return cost 

def main():
    path = 'xxx/data/Linear/'
    x,y = load_data(path)

    #linear model using sklearn
    if 1:
        lr = linear_model.LinearRegression()
        lr.fit(x,y)

        theta = lr.coef_
        bias = lr.intercept_
        rsquare = lr.score(x, y)

        print("theta: " + str(theta))
        print("bias: " + str(bias))
        print("r-square: " + str(rsquare))


        x_mat = np.mat(x)
        y_mat = np.mat(y)

        h = x_mat * np.mat(theta).transpose() + bias
        plt.plot(h,'r--')
        plt.plot(y,'b-')
        plt.ylabel("h: red, y: blue")
        plt.xlabel("m")
        plt.title("Best Fit")
        plt.show()

    #linear model using tensor flow
    if 1:
        x_train = standardization(np.mat(x), 0)
        #x_train = standardization(np.mat(x), 1)
        y_train = standardization(np.mat(y).transpose(),0)
        print np.shape(x_train)
        print np.shape(y_train)

        m,n = np.shape(x_train)

        rate = 0.002
        epoch = 180

        X = tf.placeholder("float")  
        y = tf.placeholder("float") 
        W = tf.Variable(tf.ones([2,1])) 
        b = tf.Variable(tf.random_normal([1])) 

        all_cost = []
        all_rsquare = []

        h = tf.add(tf.matmul(X,W),b)
        loss = tf.reduce_mean(tf.pow(h-y,2)) / 2 
        optimizer = tf.train.GradientDescentOptimizer(rate).minimize(loss)  

        init = tf.global_variables_initializer()

        # Second part: launch the graph
        sess = tf.Session()  
        sess.run(init)  

        for i in range(epoch):
            sess.run(optimizer,{ X:x_train, y:y_train }) 
            all_cost.append(cost(x_train, y_train, sess.run(W), sess.run(b)))
            all_rsquare.append(r_square(x_train, y_train, sess.run(W), sess.run(b)))

        W = sess.run(W)
        b = sess.run(b)
        rsquare = r_square(x_train, y_train, W, b)
        print("weights: ", W.transpose())
        print("bias: ", b)
        print("r-square: ", rsquare)

        plt.plot(all_cost,'k-', label='line 1', linewidth=2)
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("Cost Function")
        plt.show()

        plt.plot(all_rsquare,'k-', label='line 1', linewidth=2)
        plt.ylabel("r-square")
        plt.xlabel("epoch")
        plt.title("r-square")
        plt.show()

        h = x_train * np.mat(W) + b
        plt.plot(h,'r--')
        plt.plot(y_train,'b-')
        plt.ylabel("h: red, y: blue")
        plt.xlabel("m")
        plt.title("Best Fit")     
        plt.show()


if __name__ == "__main__":
    main()

