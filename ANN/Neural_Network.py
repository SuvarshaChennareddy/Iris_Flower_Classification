import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Layer:

    #Initialize weights and bias
    def __init__(self, num_inputs, num_outputs, isLinear = False):
        self.nin = num_inputs
        self.nout = num_outputs
        self.weights = np.random.uniform(low=-1, high=1, size=(self.nin, self.nout))
        self.bias = np.random.uniform(low=-1, high=1, size=(self.nout, 1))

        #Attribute to set the layer to be purely linear (no activation function)
        self.linear = isLinear

    #Implementing the ReLU activation function and its derivative
    @staticmethod
    def relu(x):
        return np.maximum(0,x)

    @staticmethod
    def relu_derivative(a):
        return np.where(a > 0, 1, 0)
    
    '''
    Additional activation functions and their derivatives

    @staticmethod
    def sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def sigmoid_derivative(a):
        return a*(1-a)

    @staticmethod
    def celu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x/alpha) - 1))

    @staticmethod
    def celu_derivative(a, alpha=1.0):
        return np.where(a > 0, 1, alpha * np.exp(a/alpha))
    '''

    #Compute the outputs of the layer
    def eval(self, inputs):
        outputs = self.weights.transpose() @ inputs + self.bias
        if(not self.linear):
            outputs = Layer.relu(outputs)
        return outputs

    #Compute the backpropagated gradient at the layer (gradient of the cost wrt inputs of the layer)
    def eval_gradient(self, outputs, backprop_gradient):
        if(not self.linear):   
            a_dash = Layer.relu_derivative(outputs)
            return (self.weights @ backprop_gradient) * a_dash
        else:
            return (self.weights @ backprop_gradient)
        
        
#Implementation of a simple neural network model for classification
class Classification_Neural_Network:
    def __init__(self):
        self.layers = []
        self.outputs = []
        self.gradients = []

    #Method to add layer
    def add_layer(self, layer):
        self.layers.append(layer)

    #Implementing softmax
    @staticmethod
    def softmax(inputs):
        max_elem = np.max(inputs)
        return np.exp(inputs - max_elem)/np.sum(np.exp(inputs - max_elem))
    
    #Implementing forward pass
    def forward(self, inputs):
        #To store the inputs and outputs of each layer. 
        self.outputs = []
        self.outputs.append(inputs)
        for layer in self.layers:
            self.outputs.append(layer.eval(self.outputs[-1]))

        #Return softmax of outputs of final layer
        return Classification_Neural_Network.softmax(self.outputs[-1])

    #Compute the categorical crossentropy
    @staticmethod
    def categorical_crossentropy(y_true, y_predicted):
        return -np.dot(y_true, np.log(y_predicted))

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
                #Storing the gradients for backpropagation
                self.gradients = []

                #Computing the outputs
                probs = self.forward(X[i])
                total_loss  += Classification_Neural_Network.categorical_crossentropy(y[i], probs.reshape((probs.size)))

                #Computing the output layer's gradient (derivative of softmax + categorical crossentropy loss wrt output of last layer)
                self.gradients.append(probs - y[i].reshape((y[i].shape[0], 1)))

                #If the last layer isn't linear, incorporate the gradient of the activation 
                if(not self.layers[-1].linear):
                    a_dash = Layer.relu_derivative(self.outputs[-1])
                    self.gradients[0] *= a_dash

                #Backpropagation (storing the gradients at each layer)
                for  l in range(len(self.layers) - 2, -1, -1):
                    self.gradients.append(self.layers[l+1].eval_gradient(
                        self.outputs[l+1], self.gradients[-1]))

                #Using the stored gradients to update the weights and bias
                for  l in range(len(self.layers) - 1, -1, -1):
                    self.layers[l].weights -= learning_rate * (self.outputs[l] @ self.gradients[len(self.layers) - l - 1].transpose())
                    self.layers[l].bias -= learning_rate * self.gradients[len(self.layers) - l - 1]

            #Append the average loss of the epoch          
            loss.append(total_loss/y.shape[0])

        #Plotting training loss
        plt.plot(loss)
        plt.ylabel("Binary Crossentropy")
        plt.xlabel("Training iterations")
        plt.title("Training loss")
        plt.show()

    #Compute accuracy over test dataset
    def accuracy(self, test_inputs, test_labels):
        acc = 0
        for i in range(test_labels.shape[0]):
            predicted_label = np.argmax(self.forward(test_inputs[i]))
            acc += (np.argmax(test_labels[i]) == predicted_label)
        acc /= test_labels.shape[0]
        return acc

