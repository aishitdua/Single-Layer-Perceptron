# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 22:31:00 2019

@author: DELL
"""
import pandas as pd        # for file reading
import numpy as np        # for arrays and calculations
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # for graphical representations
np.set_printoptions(suppress=True) # to view numbers not as powers of e but as floating points
class Perceptron():
    def __init__(self):
        self.syn_weights = np.random.rand(23,1)

    def threshold(self, x):
        if x>0:
            return 1
        else:
            return 0

    def draw(self,y):
        x=[]
        for i in range(100):
            x.append(i)
        plt.plot(x, y, linewidth=2.0)
    def train(self, inputs, real_outputs, its, lr):

        delta_weights = np.zeros((23,1600))
        errorsum=[]
        for iteration in (range(its)):
            error=0
            for i in range(1599):
                z = np.dot(inputs, self.syn_weights) # dot product
                activation = self.threshold(z[i])
                cost_prime = (activation - real_outputs[i])
                if cost_prime!=0:
                    error+=1
                for n in range(23):
                    delta_weights[n][i] = cost_prime * inputs[i][n] * lr               
                delta_avg = np.average(delta_weights)
                for n in range(23):
                    self.syn_weights[n] = self.syn_weights[n] -delta_weights[n][i]
            errorsum.append(error)
        self.draw(errorsum)

    def results(self, inputs):
        return self.threshold(np.dot(inputs, self.syn_weights))
data=pd.read_csv('dataset.csv') # reading the file
arr=data.values
y=arr[:,23]
data=data/data.max() # normalizing the data i.e. every index now has a value between 0 and 1
x=arr[:,0:23]
x_train, x_test, y_train, y_test=train_test_split(x,y,stratify=y,test_size=0.2) # divinding test and training set
lr = .5 #Learning Rate
epochs = 100
perceptron = Perceptron() # Initialize a perceptron with object of class Perceptron

perceptron.train(x_train, y_train, epochs, lr) # Train the perceptron

results = []
for i in (range(len(x_test))):
    run = x_test[i]
    trial = perceptron.results(run)
    results.append(trial)
acc=0
for i in range(len(y_test)):
    if results[i]==y_test[i]:
        acc+=1
print("accuracy=",(acc*100/len(y_test)))
