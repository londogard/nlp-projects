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
    
def createHistoryData(X, Y, days):
    X = [[X[j] for j in range(i, i+days)] for i in range(0, len(X)-days)]
    Y = Y[days:]
    return X, Y