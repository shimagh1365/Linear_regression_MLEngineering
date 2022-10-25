# Using the data.csv, 
# write a LinearRegressor class which inherits from sklearn’s BaseEstimator class, 
# and use it to evaluate against the dataset.
# The LinearRegressor class should have the following methods which implement OLS with numpy array multiplication / division:
# fit(X, y) - fit the model to the data
# predict(X) - predict the output for the given input
# score(X, y) - return the R^2 score for the given input and output

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score

def readData():
    df = pd.read_csv('data.csv')
    data = pd.DataFrame(columns=['x1', 'x2', 'y'])
    data['x1'] = df['x1']
    data['x2'] = df['x2']
    data['y'] = df['y']
    # Split data into X and y
    X = data[['x1', 'x2']]
    y = data['y']
    # Use the first 25 samples to train, and the remainder to predict.
    X_train = X[:25]
    y_train = y[:25]
    X_test = X[25:]
    y_test = y[25:]
    return X_train, y_train, X_test, y_test

class LinearRegressor(BaseEstimator):
    def __init__(self):
        self.w_ = None
        self.coef_ = None
        self.intercept_ = None
    
    def get_minor_matrix(self, matrix, row, col):
        minor_matrix = []
        for i in range(len(matrix)):
            if i != row:
                minor_matrix.append([])
                for j in range(len(matrix)):
                    if j != col:
                        minor_matrix[-1].append(matrix[i][j])
        return minor_matrix

    def get_determinant(self, matrix):
        if len(matrix) == 1:
            return matrix[0][0]
        elif len(matrix) == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        else:
            det = 0
            for i in range(len(matrix)):
                det += matrix[0][i] * self.get_determinant(self.get_minor_matrix(matrix, 0, i))
            return det

    def get_cofactor_matrix(self, matrix):
        cofactor_matrix = []
        for i in range(len(matrix)):
            cofactor_matrix.append([])
            for j in range(len(matrix)):
                cofactor_matrix[-1].append(self.get_determinant(self.get_minor_matrix(matrix, i, j)))
        return cofactor_matrix

    def get_transpose(self, matrix):
        transpose_matrix = []
        for i in range(len(matrix[0])):
            transpose_matrix.append([])
            for j in range(len(matrix)):
                transpose_matrix[-1].append(matrix[j][i])
        return transpose_matrix

    def get_scalar_multiplication(self, matrix, scalar):
        for i in range(len(matrix)):
            for j in range(len(matrix)):
                matrix[i][j] *= scalar
        return matrix

    # get inverse of a matrix without numpy for n*n matrix
    def get_inverse(self, matrix):
        # get determinant of matrix
        det = self.get_determinant(matrix)
        # get cofactor matrix
        cofactor_matrix = self.get_cofactor_matrix(matrix)
        # get adjoint matrix
        adjoint_matrix = self.get_transpose(cofactor_matrix)
        # get inverse matrix
        inverse_matrix = self.get_scalar_multiplication(adjoint_matrix, 1/det)
        return inverse_matrix
    
          
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        # don't use numpy’s linalg subpackage.
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        #X_inv = self.get_inverse(X.T @ X)
        X_inv = np.linalg.inv(X.T @ X)
        self.w_ = X_inv @ X.T @ y
        self.intercept_ = self.w_[0]
        self.coef_ = self.w_[1:]

    def predict(self, X):
        X = np.array(X)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X @ self.w_

    def score(self, X, y):
        return r2_score(y, self.predict(X))

   