
import math
import numpy as np
import matplotlib.pyplot as plt

######### Generic Layer Object #################

class Layer(object):
    def __init__(self, **kwargs):
        self.attrDict = {'input_dim'  : 'n', 
                         'output_dim' : 'm', 
                         'output'     : 'O'}
        self.setAttr(**kwargs)
        
    def setAttr(self, **kwargs):
        if kwargs is not None:
            for key, value in kwargs.iteritems():
                if key in self.attrDict:
                    setattr(self, self.attrDict[key], value)

######### Input Layer Object #################                    
                    
class InputLayer(Layer):
    def __init__(self, input_dim, bias = True, **kwargs):
        super(InputLayer, self).__init__(**kwargs)
        
        self.type_ = 'input'
        self.n = input_dim
        self.bias = bias
        self.composed = False
        self.loss_dbydx = None
        
    def compose(self, **kwargs):
        super(InputLayer, self).setAttr(**kwargs)

        try: 
            assert self.n > 0
            
            self.m = self.n
            if self.bias: self.m = self.m + 1
            self.W = np.identity(self.n, dtype=float) 
            self.O = np.zeros(self.m, dtype=float)

            self.composed = True
            return self

        except AssertionError: print("Error: The Layer dimensions should be greater than 0")

    def forward(self, X):
        
        self.O = np.dot(self.W, X)
        if self.bias: self.O = np.insert(self.O, 0, 1)
        return self.O
    
######### Dense Layer Object #################    

class Dense(Layer):
    def __init__(self, output_dim, **kwargs):
        super(Dense, self).__init__(**kwargs)
        
        self.type_ = 'dense'
        self.m = output_dim
        self.composed = False
        self.loss_dbydx = None
        
    def compose(self, **kwargs):
        super(Dense, self).setAttr(**kwargs)

        try: 
            assert self.n > 0 and self.m > 0

#             self.W = np.zeros((self.m, self.n), dtype=float) 
            self.W = np.random.randn(self.m, self.n)   
            self.O = np.zeros((self.m), dtype=float)
        
            self.W_dot = np.zeros((self.m), dtype=float)           

            self.composed = True
            return self

        except AssertionError: print("Error: The Layer dimensions should be greater than 0")

    def forward(self, X):
        
        self.O = np.dot(self.W, X)
        return self.O
    
    def backward(self, X):
        return self.W
    
######### Activation Layer Object #################    
    
class Activation(Layer):
    
    def __init__(self, func_type, bias = True, **kwargs):
        super(Activation, self).__init__(**kwargs)
        
        self.funcDict = {'sigmoid' : Sigmoid()}
        
        self.type_ = 'activation'
        self.activation = self.funcDict[func_type]
        self.f    = np.vectorize(self.activation.f)
        self.fdot = np.vectorize(self.activation.fdot)
        self.bias = bias
        self.loss_dbydx = None
        
    def compose(self, **kwargs):
        super(Activation, self).setAttr(**kwargs)
        try: 
            assert self.n > 0
            
            self.m = self.n
            if self.bias: self.m = self.m + 1
            
            self.O = np.zeros(self.m)
                
            return self
        
        except AssertionError: print("Error: The Layer dimensions should be greater than 0")

    def forward(self, X):
        self.O = self.f(X)
        if self.bias:  self.O = np.insert(self.O, 0, 1)
        return self.O
    
    def backward(self, X):
        return np.diag(self.fdot(X))
    
######### Activation function Object #################    
    

class Sigmoid:
    def f(self, x):
        return 1 / (1 + math.exp(-x))
    def fdot(self, x):
        sigma = self.f(x)
        return sigma * (1-sigma)    
    

######### Loss #################    
            
class Loss:
    def __init__(self, func_type):
        
        self.funcDict = {'squared_error' : SquaredErr()}
        
        self.f    = self.funcDict[func_type].f
        self.fdot = self.funcDict[func_type].fdot

class SquaredErr:
    def f(self, y_pred, y_true):
        return np.sum((y_pred-y_true)**2)/2
    def fdot(self, y_pred, y_true):
        return (y_pred-y_true)
    
######### Metric #################    
    
class Metric:
    def __init__(self, func_type):
        
        self.funcDict = {'mean_squared_error' : MeanSquaredError()}
        
        self.f    = self.funcDict[func_type].f

class MeanSquaredError:
    def f(self, y_pred, y_true):
        return np.sum((y_pred - y_true)**2)  
    
######### Model Object #################    

class Model:
    'Neural Network class for classification'
    
    def __init__(self):
        
        self.layers = []
        self.O = []
        self.composed = False
        
    def add(self, layer):
                
        if not self.layers:  assert layer.n > 0
        self.layers.append(layer)
                
        self.composed = False
        return self
                
    def compose(self, loss = Loss('squared_error'), metric = Metric('mean_squared_error')):
        try: 
            assert not not self.layers
            
            self.loss = loss
            self.metric = metric
            
            self.n = self.layers[0].n
            dim = self.n
            self.layers = [InputLayer(input_dim = dim)] + self.layers
            for l in self.layers: 
                dim = l.compose(input_dim = dim).m
            
            self.m = dim
            self.composed = True
            return self
        
        except AssertionError: print("Error: The model has no layers. Use Model.add()")
    
    def forward(self, x):
        try: 
            assert self.composed == True
            
            self.O = [self.layers[0].forward(x)]
            for l in self.layers[1:]:
                self.O.append(l.forward(self.O[-1]))
#                 print self.O
            return self.O[-1]
        
        except AssertionError: print("Error: The model has not been composed. Use Model.compose()")
            
    def predict(self, X):
        if X.ndim == 1:
            if X.size == self.n: return self.forward(X)
            else:
                Y_pred = np.empty((X.shape[0],self.m))
            for i in range(Y_pred.shape[0]): 
                Y_pred[i] = self.forward(X[i])
                
        elif X.ndim == 2:
            Y_pred = np.zeros((X.shape[0],self.m), dtype=float)
            for i in range(Y_pred.shape[0]): 
                Y_pred[i] = self.forward(X[i])
        else: print('Error: Input Error')
            
        return Y_pred
    
    def evaluate(self, X, Y):
        Y_pred = self.predict(X)
        return self.metric.f(Y_pred, Y)
    
    def backProp(self, x, y):
        try:
            assert self.composed == True 
            
            self.layers[-1].loss_dbydx = self.loss.fdot(self.forward(x), y)
            for l in reversed(range(len(self.layers))[:-1]):
#                 print('l = '+ str(l))
#                 print('loss = \n' + str(self.layers[l+1].loss_dbydx))
#                 print('back = \n' + str(self.layers[l+1].backward(self.layers[l].O)))
                self.layers[l].loss_dbydx = np.einsum('i,ij->j', 
                                                      self.layers[l+1].loss_dbydx, 
                                                      self.layers[l+1].backward(self.layers[l].O))
                
                if self.layers[l].type_ == 'activation' and self.layers[l].bias == True:
                    self.layers[l].loss_dbydx = self.layers[l].loss_dbydx[1:]
                
            for l in range(len(self.layers)):
                if self.layers[l].type_ == 'dense':
                    self.layers[l].W_dot = np.outer(self.layers[l].loss_dbydx, self.layers[l-1].O)
#                     print('w_dot = \n' + str(self.layers[l].W_dot))
            
        except AssertionError: print("Error: The model has not been composed and propogated at least once. \
                                     Use Model.compose() and Model.forward()")
            
    def updateNetwork(self, alpha):
        for l in self.layers:
            if l.type_ == 'dense':
                l.W = l.W - alpha * l.W_dot
                
    def fit(self, X, Y, epoch, alpha):
        err = []
        for e in range(epoch):
            for x,y in zip(X,Y):
#                 print('x = ' + str(x) + 'y = ' + str(y))
                self.backProp(x,y)
                self.updateNetwork(alpha)
                err.append(self.evaluate(X,Y))
        return err
     
