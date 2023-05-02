import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ANN.Neural_Network import *
from LogisticRegression.Logistic_Regression import *

df = pd.read_csv("./Data/Iris.csv")

#Visualizing the data
sns.pairplot(df[df.columns[1:]], hue="Species")
plt.show()

X = df[df.columns[1:-1]].to_numpy()
X = X.reshape((X.shape[0], X.shape[1], 1))

#For Logistic Regression Models
y1 = np.zeros((X.shape[0]))
y2 = np.zeros((X.shape[0]))
y3 = np.zeros((X.shape[0]))

#For Neural Network
y = np.zeros((X.shape[0], 3))

#Logistic Rgression: Preparing the labels for one vs all binary classification
#Neural Networl: One hot encoding the labels
for i in range(X.shape[0]):
    species = df.at[i, "Species"]
    if(species == "Iris-setosa"):
        y1[i] = 1
        y[i, 0] = 1
        
    elif(species == "Iris-versicolor"):
        y2[i] = 1
        y[i, 1] = 1
        
    elif(species == "Iris-virginica"):
        y3[i] = 1
        y[i, 2] = 1


#Splitting the dataset to training and testing datasets
X_test = X[130:]
y1_test = y1[130:]
y2_test = y2[130:]
y3_test = y3[130:]
y_test = y[130:]

X = X[0: 130]
y1 = y1[0: 130]
y2 = y2[0: 130]
y3 = y3[0: 130]
y = y[0: 130]


#Model 1 will predict whether the species is Iris-setosa
LR1 = Logistic_Regression(X.shape[1])
print("Model 1's accuracy before training: ", LR1.accuracy(X_test, y1_test))
LR1.train(X, y1, 0.001, 250)
print("Model 1's accuracy after training: ", LR1.accuracy(X_test, y1_test))

#Model 2 will predict whether the species is Iris-versicolor
LR2 = Logistic_Regression(X.shape[1])
print("Model 2's accuracy before training: ", LR2.accuracy(X_test, y2_test))
LR2.train(X, y2, 0.001, 250)
print("Model 2's accuracy after training: ", LR2.accuracy(X_test, y2_test))

#Model 3 will predict whether the species is Iris-virginica
LR3 = Logistic_Regression(X.shape[1])
print("Model 3's accuracy before training: ", LR3.accuracy(X_test, y3_test))
LR3.train(X, y3, 0.001, 250)
print("Model 3's accuracy after training: ", LR3.accuracy(X_test, y3_test))

#Neural network for classification of species
nn = Classification_Neural_Network()
nn.add_layer(Layer(X.shape[1], 10))
nn.add_layer(Layer(10, 10))
nn.add_layer(Layer(10, y.shape[1], True))
print("Accuracy before training: ", nn.accuracy(X_test, y_test))
nn.train(X, y, 0.001, 100)
print("Accuracy after training: ", nn.accuracy(X_test, y_test))
