import numpy as np

def splitData(X, Y, amount):
    X_train, X_test = np.split(X, [int(amount*len(X))])
    Y_train, Y_test = np.split(Y, [int(amount*len(Y))])
    return X_train, X_test, Y_train, Y_test

def printResults(pred, Y_test, X_test):
    for i in range(len(pred)):
        print("Open: ", X_test[i])
        print("Pred: ", pred[i])
        print("RealValue: ", Y_test[i])
        print("-----------------------------")
    
def createHistoryData(X, Y, days):
    X = [X[i:i+days] for i in range(0, len(X)-days)]
    Y = Y[days:]
    return X, Y