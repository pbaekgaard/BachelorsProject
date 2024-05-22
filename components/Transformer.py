from sktime.base import load
import os
import pandas as pd
from pycatch22 import catch22_all as Catch22
import numpy as np
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from scipy.fft import fft
from scipy.signal import welch
from scipy.stats import pearsonr
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import entropy


def calculate_rms(data):
    """
    Calculate Root Mean Square (RMS) for each axis of the data.

    Parameters:
    - data: numpy array with shape (n_axes, n_samples), data for each axis

    Returns:
    - rms: numpy array with shape (n_axes,), RMS for each axis
    """
    rms = np.sqrt(np.mean(data**2, axis=1))
    return rms


def calculate_zero_crossing_rate(data):
    """
    Calculate zero-crossing rate for each axis of the data.

    Parameters:
    - data: numpy array with shape (n_axes, n_samples), data for each axis

    Returns:
    - zero_crossing_rate: numpy array with shape (n_axes,), zero-crossing rate for each axis
    """
    zero_crossing_rate = np.zeros(data.shape[0])
    for i in range(data.shape[0]):  # Loop over axes
        zero_crossing_rate[i] = np.sum(np.diff(np.sign(data[i])) != 0) / (2 * len(data[i]))
    return zero_crossing_rate


def calculate_cross_correlation(accel_data, gyro_data):
    """
    Calculate cross-correlation between accelerometer and gyroscope data.

    Parameters:
    - accel_data: numpy array with shape (3, n_samples), accelerometer data for x, y, and z axes
    - gyro_data: numpy array with shape (3, n_samples), gyroscope data for x, y, and z axes

    Returns:
    - cross_corr_matrix: numpy array with shape (3, 3), cross-correlation matrix
    """
    cross_corr_matrix = np.zeros((3, 3))
    for i in range(3):  # Loop over accelerometer axes
        for j in range(3):  # Loop over gyroscope axes
            cross_corr_matrix[i, j] = np.corrcoef(accel_data[i], gyro_data[j])[0, 1]
    return cross_corr_matrix


def Transform(data_sequences):
    print("Transforming data using Catch22")
    data = []
    for idx, dataSeq in enumerate(data_sequences):
        print(f"Transforming sequence {idx+1}/{len(data_sequences)}")
        dataSeq = dataSeq.T[::20].T
        dataSeq = dataSeq[0:3]
        std = np.std(dataSeq.T, axis=0)
        skewness = pd.DataFrame(dataSeq.T).skew()
        kurtosis = pd.DataFrame(dataSeq.T).kurtosis()
            
        # print(f"Mean: {mean}\n Std: {std}\n Skewness: {skewness}\n Kurtosis: {kurtosis}")
        feature_matrix = np.column_stack((std, skewness, kurtosis))
        data.append(feature_matrix.T.flatten())
    print("Transformation complete!")
    return data
