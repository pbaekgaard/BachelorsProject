import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def Transform(data_sequences):
    print("Transforming data using Catch22")
    data = []
    for idx, dataSeq in enumerate(data_sequences):
        k = 20
        print(f"Transforming sequence {idx+1}/{len(data_sequences)}")
        dataSeq = dataSeq.T[::20].T
        accel = np.mean(dataSeq[:3], axis=0)
        accelSorted = np.sort(accel)
        gyro = np.mean(dataSeq[3:], axis=0)
        gyroSorted = np.sort(gyro)
        kMin = accelSorted[:k]
        kMax = accelSorted[-k:]
        kMinGyro = gyroSorted[:k]
        kMaxGyro = gyroSorted[-k:]

        dataOut = np.column_stack((kMin, kMax, kMinGyro, kMaxGyro))
        data.append(dataOut.T)
    print("Transformation complete!")
    return data
