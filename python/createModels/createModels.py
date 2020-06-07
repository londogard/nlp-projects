import pandas as pd
from sklearn.linear_model import LinearRegression

def baseline(stockData):
    pd_data = pd.read_csv(stockData)
    X = pd_data["Open"]
    Y = pd_data["High"].values
    X = X.values.reshape(-1,1)
    X_train, X_test, Y_train, Y_test = splitData(X, Y, 0.75)
    regr = LinearRegression()
    regr.fit(X_train, Y_train)
    pred = regr.predict(X_test)
    printResults(pred, Y_test, X_test)

def splitData(X, Y, amount):
    X_train =  X[:round(len(X)*amount)]
    X_test =  X[round(len(X)*amount):]
    Y_train =  Y[:round(len(Y)*amount)]
    Y_test =  Y[round(len(Y)*amount):]
    return X_train, X_test, Y_train, Y_test

def printResults(pred, Y_test, X_test):
    for i in range(len(pred)):
        print("Open: ", X_test[i])
        print("Pred: ", pred[i])
        print("RealValue: ", Y_test[i])
        print("-----------------------------")

if __name__ == "__main__":
    baseline("../generateStockData/data/EVO.ST_2020-03-01_2020-05-31.csv")