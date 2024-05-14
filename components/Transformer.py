import pandas as pd
import numpy as np


def Transform(data_sequences):
    print("Transforming data using Catch22")
    data = []
    for idx, dataSeq in enumerate(data_sequences):
        k = 3
        print(f"Transforming sequence {idx+1}/{len(data_sequences)}")
        dataSeq = dataSeq.T[::20].T
        accel = np.mean(dataSeq[:3], axis=0)
        accelSorted = np.sort(accel)

        kMin = accelSorted[:k]
        kMax = accelSorted[-k:]
        print(f"kMin: {kMin}")
        print(f"kMax: {kMax}")


        dataOut = np.column_stack((kMin, kMax))
        data.append(dataOut.T)
    print("Transformation complete!")
    return data
