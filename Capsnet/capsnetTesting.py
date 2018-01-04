import numpy as np

def getAcc(y_test,y_pred):
    return  np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0]
