from sktime.base import load
import os
import pandas as pd
from pycatch22 import catch22_all as Catch22


def Transform(data_sequences):
    print("Transforming data using Catch22")
    data = []
    for idx, dataSeq in enumerate(data_sequences):
        print(f"Transforming sequence {idx+1}/{len(data_sequences)}")
        flattenedData = dataSeq.flatten()
        transformedDataRaw = Catch22(flattenedData)
        transformedData = transformedDataRaw["values"]
        data.append(transformedData)

    print("Transformation complete!")
    return data
