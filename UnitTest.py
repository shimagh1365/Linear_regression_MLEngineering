from Regressor import LinearRegressor, readData
import unittest
from sklearn.linear_model import LinearRegression
import numpy as np
# write unit tests for your code
# use the data.csv file to test your code
# use the unittest module to write unit tests
# use the LinearRegressor class to evaluate  

class TestLinearRegressor(unittest.TestCase):
    def setUp(self):
        self.X_train, self.y_train, self.X_test, self.y_test = readData()
        self.my_lr = LinearRegressor()
        self.my_lr.fit(self.X_train, self.y_train)
       
        self.sklearn_lr = LinearRegression()
        self.sklearn_lr.fit(self.X_train, self.y_train)
    
    def test_fit(self):
        self.assertTrue(np.allclose(self.my_lr.coef_, self.sklearn_lr.coef_), 'Model run with LinearRegressor and LinearRegression do not have the same coefficients')
        self.assertTrue(np.allclose(self.my_lr.intercept_, self.sklearn_lr.intercept_), 'Model run with LinearRegressor and LinearRegression do not have the same intercepts')
    
    def test_predict(self):
        self.assertTrue(np.allclose(self.my_lr.predict(self.X_test), self.sklearn_lr.predict(self.X_test)), 'Model run with LinearRegressor and LinearRegression do not have the same predictions')
        
    
    def test_score(self):
        self.assertTrue(np.allclose(self.my_lr.score(self.X_test, self.y_test), self.sklearn_lr.score(self.X_test, self.y_test)), 'Model run with LinearRegressor and LinearRegression do not have the same scores')
           

def runTest():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinearRegressor)
    unittest.TextTestRunner(verbosity=2).run(suite)
