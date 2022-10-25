# Linear Regressor using numpy arrays
Our goal in this project is to implement linear regression using arrays in the Numpy library.
## Overview
This work is divided into two main parts:
The first part includes the implementation of **linear regression** class which inherits from sklearn’s BaseEstimator class, and use it to evaluate against the dataset and performing various unit tests on it.
The second part includes creating an **Endpoint** to predict and generate Dockerfile for the project.
## Usage
The entry point of the file project is `main.py`. When this file is executed, the server is started and ready to receive input and perform prediction. If you want to do other features of the project, you can press the `Ctrl+D` key and access other features.
To apply linear regression on the dataset, the `Regressor.py` file must be executed.
Performing the necessary tests on linear regression is done by the file `UnitTest.py`
Building a Python program that exposes an HTTP endpoint for the prediction method can also be done by the file `endpoint.py`
By running this file, Flask server waits for input on local machine and `port 8000` and then performs prediction.
## Dependencies
- numpy
- pandas
- scikit-learn
- unittest2
- Flask