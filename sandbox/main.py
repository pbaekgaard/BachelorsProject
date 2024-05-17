import numpy as np
from pycatch22 import catch22_all

# Generate random multivariate time series data
data = np.random.randn(6, 300)  # 300 time points, 6 variables

# Extract Catch22 features for each time series
catch22_features = []
for series in data.T:  # Transpose data to iterate over each variable
    features = catch22_all(series)
    catch22_features.append(features)

# Convert the list of feature arrays into a 2D NumPy array
catch22_features = np.array(catch22_features)

print("Catch22 features shape:", catch22_features.shape)
# print("Extracted Catch22 features for the first variable:", catch22_features)
