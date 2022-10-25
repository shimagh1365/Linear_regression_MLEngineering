# in this file we want run Regressor, UnitTest and endpoint respectively
from time import sleep
from Regressor import LinearRegressor, readData
from UnitTest import runTest
import endpoint
import pandas as pd

# run the app
if __name__ == '__main__':
    print('Server run on port 8000.')
    print('If you want to stop the server, please press Ctrl+C.')
    endpoint.runServer()
    while(True):
        print("1. Run Regressor")
        print("2. Run UnitTest")
        print("3. Run endpoint")
        print("4. Exit")
        print("Enter your choice: ")
        choice = int(input())
        if choice == 1:
            print("Regressor is running")
            print("Reading data...")
            X_train, y_train, X_test, y_test = readData()
            sleep(2)
            print("Training model...")
            model = LinearRegressor()
            model.fit(X_train, y_train)
            sleep(2)
            print("Predicting...")
            y_pred = model.predict(X_test)
            sleep(2)
            # Output the predictions in predictions.csv
            print('Writing predictions to predictions.csv...')
            df = pd.DataFrame(columns=['y'])
            df['y'] = y_pred
            df.to_csv('predictions.csv', index=False) 
            print('Predictions written successfully') 
            print('Regressor Done.')
           
        elif choice == 2:
            print("UnitTest is running\n")
            runTest()
            print("\nUnitTest Done.")
            
        elif choice == 3:
            print("endpoint is running")
            endpoint.runServer()

        elif choice == 4:
            break
        else:
            print("Invalid choice")