""" Import numpy for handling calculations, pandas for data handling and matplotlib for plotting the data"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

""" Initialize the dataframe(df) that will handle the data from the file handled by pandas"""
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

""" Test the data that was put into the dataframe by printing the last 10 values"""
df.tail(n = 10)

"""Store the values of the first 100 items together with their names"""
y = df.iloc[0:100, 4].values

""" using numpy .where() fnct to provide all Iris-setosa with the value of -1, and 1 to the rest"""
y = np.where(y == "Iris-setosa", -1, 1)

"""Initializa  X with the first 100 items, with two attributes 0 and 2"""
X = df.iloc[0:100, [0, 2]].values

"""Use the plt.scatter() to map the items onto a grid"""
plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="setosa")

plt.scatter(X[50:100, 0], X[50:100, 1], color="blue", marker="x", label="versicolor")

""" Designate xlabel, ylabel, position of the guide and finally show the figure."""
plt.xlabel("Petal length")
plt.ylabel("Sepal lenght")
plt.legend(loc = "upper left")
plt.show()

class perceptron(object):
  def __init__(self, eta = 0.01, n_iter = 10):
    self.eta = eta
    self.n_iter = n_iter
    
  def fit(X, y):
    self.w_ = np.zeros(1 + X.shape[1])
    self.errors_ = []
    for _ in range(self.n_iter):
      error = 0
      for xi, target in zip(X, y):
        update = self.eta * (target - self.predict(xi))
        self.w_[1:] += update * xi
        self.w_[0] += update
        error += int(update != 0.0)
      self.errors_.append(error)
    return self
	
	def predict(self, X):
		return np.where(self.net_input(X) >= 0.0, 1, -1)
	
	def net_input(self, X):
		return np.dot(X, self.w_[1:]) + self.w_[0]
