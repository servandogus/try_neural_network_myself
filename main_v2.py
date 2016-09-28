# -*- coding: utf-8 -*-

import random
import numpy as np

class Network:
    ''' 
    A neural network.
    See the url https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    and \picture backpropagationcalculations.jpg.
    '''
    
    def __init__(self, list_layers):
        ''' Initialise un reseau de neurones a partir de la liste fournit.
        list_layers[i] donne le nombre de neurones de la i-ème couche.
        '''
        self.layers_desc = list_layers
        self.layers = [np.zeros((1,y))[0] for y in list_layers] # np.zeros(shape)[0] gives a vector
        self.net_layers = list(self.layers)
        self.weights = [np.random.randn(x, y) for x, y in zip(list_layers[:-1],list_layers[1:])]
        # For e.g sizes = [4,5,1], then:
        # zip(sizes[:-1],sizes[1:]) yied:
        #	(4,5)
        #	(5,1)        
                
        self.biaises = [np.random.randn(1, y)[0] for y in list_layers[1:]]
    
    def f(self, x):
        return 1/(1 + np.exp(-x))
        
    def calculate(self, x):
        ''' give the ouput by network for the input x.
        x should be a vector of self.layers[0] size.
        The output will be a vector or scalar, like the last layer of the network.
        '''
        self.layers[0] = x
        for i in range(1, len(self.layers)):
            self.net_layers[i] = np.dot(self.layers[i-1], self.weights[i-1])[0] + self.biaises[i-1]
            self.layers[i] = self.f(self.net_layers[i])
        return self.layers[-1:]
    
    def error(self, expected):
        res = 0
        for i in range(len(self.layers[-1:])):
            res += 0.5 * (expected[i] - self.layers[-1:][i])**2
        return res
        
    def backprop(self, expected, eps):
        backpropagation = list(self.layers)
        del backpropagation[0]
        
        # output layer:
        for j in range(len(backpropagation[-1])):
            backpropagation[-1][j] = (self.layers[-1][j] - expected) * self.layers[-1][j] * (1 - self.layers[-1][j])

        # hidden layers :s
        for i in range(1, len(backpropagation)) :
            for j in range(len(backpropagation[-i-1])):
                backpropagation[-i-1][j] = 0
                for k in range(len(backpropagation[-i])):
                    backpropagation[-i-1][j] += backpropagation[-i][k] * self.weights[-i][j][k]
                backpropagation[-i-1][j] *= self.layers[-i-1][j] * (1 - self.layers[-i-1][j]) 
        
        for i in range(len(self.weights)):
            for j in range(len(self.weights[i])):
                for k in range(len(self.weights[i][j])):
                    self.weights[i][j][k] -= eps * backpropagation[i][k] * self.layers[i][j] 
        
        
    def learn(self, inputs, N, eps):
        for i in range(N):
            random.shuffle(inputs)
            for input in inputs:
                self.calculate(input[0])
                self.backprop(input[1], eps)    

    def check_result(self, inputs):
        res = 0        
        for i in inputs:
            if( (self.calculate(i[0]) - i[1])**2 < 0.25):
                res += 1
        print("{} / {}".format(res, len(inputs)))
        
S=[[[1, 1, 1, 1, 1, 1, 0], 0],
 [[0, 1, 1, 0, 0, 0, 0], 1],
 [[1, 1, 0, 1, 1, 0, 1], 0],
 [[1, 1, 1, 1, 0, 0, 1], 1],
 [[0, 1, 1, 0, 0, 1, 1], 0],
 [[1, 0, 1, 1, 0, 1, 1], 1],
 [[0, 0, 1, 1, 1, 1, 1], 0],
 [[1, 1, 1, 0, 0, 0, 0], 1],
 [[1, 1, 1, 1, 1, 1, 1], 0],
 [[1, 1, 1, 1, 0, 1, 1], 1]]

for i in S:
    i[0] = np.array(i[0])
    i[1] = np.array(i[1])
    i = np.array(i)
        
n = Network([7,1])
n.calculate(S[0][0])
n.learn(S, 1000, 0.01)
n.check_result(S)