import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def Transform(data_sequences):
    print("Transforming data using Catch22")
    data = []
    for idx, dataSeq in enumerate(data_sequences):
        k = 5
        print(f"Transforming sequence {idx+1}/{len(data_sequences)}")
        dataSeq = dataSeq.T[::20].T
        # ACCELEROMETER
        accel = np.mean(dataSeq[:3], axis=0)
        accelSorted = np.sort(accel)
        kMin = accelSorted[:k]
        kMax = accelSorted[-k:]

        

        dataOut = np.column_stack((kMin, kMax))
        data.append(dataOut.T)
    print("Transformation complete!")
    return data
