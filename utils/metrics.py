import numpy as np
from DDTW.fastddtw import fast_ddtw

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean()


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def CR(pred, true):
    return 1 - np.mean(np.sqrt(np.mean((((true-pred) / np.max(true,axis=0)) ** 2), axis=0)))

# Accuracy within 10% error
def ACC(pred, true):
    temp = np.abs(true-pred) / np.max(true, axis=0)
    temp[temp < 0.1] = 1
    temp[temp >= 0.1] = 1
    return np.mean(temp)

def DDTW(pred, true):
    distances = []
    for i in range(true.shape[0]):
        y1 = true[i]
        y2 = pred[i]
        distance, _ = fast_ddtw(y1, y2)
        distances.append(distance[0])
    distances = np.array(distances)
    return np.mean(distances)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)
    # ddtw = DDTW(pred,true)

    return mae, mse, rmse, mape, mspe, rse, corr
