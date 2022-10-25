from time import sleep
from Regressor import LinearRegressor
from Regressor import readData
from flask import Flask, render_template, request
import numpy as np


#• Build a Python app exposing an HTTP endpoint for the predict method

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        x1 =request.form['X1']
        # convert to float
        x1 = float(x1)
        x2 =request.form['X2']
        # convert to float
        x2 = float(x2)
        model = train()
        sleep(3)
        X = np.array([[x1, x2]])
        y = model.predict(X)
        # return 'Prediction for [{x1}, {x2}] = {y}'
        result = str.format('Prediction for [{x1}, {x2}] = {y}', x1=x1, x2=x2, y=y)
        
    return render_template('main.html', predict_y = result) 

def train():
    X_train, y_train, X_test, y_test = readData()
    model = LinearRegressor()
    model.fit(X_train, y_train)
    
    return model

def runServer():
    app.run(port=8000)
