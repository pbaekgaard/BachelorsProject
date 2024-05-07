from sktime.base import load
import os
import pandas as pd
from pycatch22 import catch22_all as Catch22
import numpy as np


def Transform(data_sequences):
    print("Transforming data using Catch22")
    data = []
    for idx, dataSeq in enumerate(data_sequences):
        print(f"Transforming sequence {idx+1}/{len(data_sequences)}")
        flattenedData = dataSeq.T
        transformedData = []
        for flatDat in flattenedData:
            features = Catch22(flatDat, True)["values"]
            transformedData.append(features)

        transformedData = np.array(transformedData)
        aggregatedData = np.mean(transformedData, axis=0)
        data.append(aggregatedData)
    print("Transformation complete!")
    return data
