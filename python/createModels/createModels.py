import pandas as pd
from sklearn.linear_model import LinearRegression
import helpFunctions as hf
import numpy as np

def baseline(stockData):
    X, Y = prepareData(stockData)
    X = X.values.reshape(-1,1)
    X_train, X_test, Y_train, Y_test = hf.splitData(X, Y, 0.75)
    regr = LinearRegression()
    regr.fit(X_train, Y_train)
    pred = regr.predict(X_test)
    hf.printResults(pred, Y_test, X_test)

def baselineWithHistory(stockData):
    X, Y = prepareData(stockData)
    X, Y = hf.createHistoryData(X.values, Y, 10)
    X_train, X_test, Y_train, Y_test = hf.splitData(X, Y, 0.75)
    regr = LinearRegression()
    regr.fit(X_train, Y_train)
    pred = regr.predict(X_test)
    hf.printResults(pred, Y_test, X_test)

def prepareData(stockData):
    pd_data = pd.read_csv(stockData)
    X = pd_data["Open"]
    Y = pd_data["High"].values
    return X, Y

if __name__ == "__main__":
    baselineWithHistory("../generateStockData/data/EVO.ST_2019-01-01_2020-05-31.csv")