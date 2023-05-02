import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Implementation of simple logistic regression model
class Logistic_Regression:

    #Initialize weights and bias
    def __init__(self, num_inputs):
        self.weights = np.random.uniform(low=-2, high=2, size=(num_inputs, 1))
        self.bias = np.random.uniform(low=-2, high=2, size=(1, 1))

    #Implementing the function
    def predict(self, inputs):
        return 1/(1+np.exp(-((self.weights.transpose() @ inputs) + self.bias)))

    #Compute the binary crossentropy loss
    @staticmethod
    def binary_crossentropy(y_true, y_predicted):
        return -(y_true * np.log(y_predicted) + (1-y_true) * np.log(1-y_predicted))

    #Train using Stochastic Gradient Descent (batch size = 1)
    def train(self, training_inputs, training_labels, learning_rate, epochs):
        
        #To keep track of the loss after every epoch
        loss = []
        for _ in range(epochs):
            total_loss = 0

            #Shuffling the dataset
            perm = np.random.permutation(len(training_labels))
            X = training_inputs[perm]
            y = training_labels[perm]

            for i in range(y.shape[0]):

                #Computing prediction
                predicted_y = self.predict(X[i])
                
                total_loss  += Logistic_Regression.binary_crossentropy(y[i], predicted_y[0, 0])

                #Calculating the gradient of the loss wrt the weights
                gradient = (predicted_y[0,0] - y[i]) * X[i]

                #Update step
                self.weights -= learning_rate * gradient
                
            #Append the average loss of the epoch    
            loss.append(total_loss/y.shape[0])

        #Plotting training loss
        plt.plot(loss)
        plt.ylabel("Categorical Crossentropy")
        plt.xlabel("Training iterations")
        plt.title("Training loss")
        plt.show()
        
    #Compute accuracy over test dataset
    def accuracy(self, test_inputs, test_labels):
        acc = 0
        for i in range(test_labels.shape[0]):
            predicted_label = 1 if self.predict(test_inputs[i])[0,0] >= 0.5 else 0
            acc += (test_labels[i] == predicted_label)
        acc /= test_labels.shape[0]
        return acc
