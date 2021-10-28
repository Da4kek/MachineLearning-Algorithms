import numpy as np

class Perceptron2():
    def __init__(self,num_features):
        self.num_features=num_features
        self.weights = np.zeros((num_features,1),dtype=np.float)
        self.bias = np.zeros(1,dtype=np.float)

    def forward(self,x):
        linear = np.dot(x,self.weights) + self.bias
        predictions = np.where(linear > 0. , 1 ,0)
        return predictions
    
    def backward(self,x,y):
        prediction=self.forward(x)
        errors = y-prediction
        return errors
    
    def train(self,x,y,epochs):
        for e in range(epochs):
            for i in range(y.shape[0]):
                errors = self.backward(x[i].reshape(1,self.num_features),y[i]).reshape(-1)
                self.weights += (errors * x[i]).reshape(self.num_features,1)
                self.bias += errors
    
    def evaluate(self,x,y):
        predictions=self.forward(x).reshape(-1)
        accuracy = np.sum(predictions==y)/y.shape[0]
        return accuracy
        