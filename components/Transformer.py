from sktime.transformations.panel.catch22 import Catch22
from sktime.base import load
import os
import pandas as pd


def Transform(data_sequences):
    print("Transforming data using Catch22")
    c22 = Catch22(
        n_jobs=-1,
        replace_nans=True,
    )
    c22.set_config(
        **{
            "backend:parallel": "joblib",  # set backend here
            "backend:parallel:params": {"n_jobs": -1},  # pass params to backed, e.g., to joblib.Parallel
        }
    )
    modelLoaded = False
    if os.path.exists("models/catch22.zip"):
        c22 = load("models/catch22")
        modelLoaded = True
        print("Model loaded!")
    else:
        print("Model not found, training new model!")
    data = []
    for dataSeq in data_sequences[:20]:
        # print(f"Transforming sequence {idx+1}/{len(data_sequences)}")
        if modelLoaded:
            transformedSequence = c22.transform(dataSeq)
        else:
            transformedSequence = c22.fit_transform(dataSeq)

        # print(transformedSequence[0:21])
        data.append(transformedSequence)
    if not os.path.exists("models"):
        os.mkdir("models")
    c22.save("models/catch22")
    print("Transformation complete!")
    return data
