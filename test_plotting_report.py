# plotting the elliptic ratio and area
# read text file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
from scipy.optimize import minimize
from scipy.optimize import brute

# Specify the file path
file_path = '/home/yungselm/Documents/IVUSimages/NARCO_119/20230809/125657/Run1/PDBHSCIO_report.txt'

# Read the data from the file into a DataFrame
df = pd.read_csv(file_path, delimiter='\t')
print(df)

# normalize lumen area and elliptic ratio
df['Lumen area (mm²)'] = df['Lumen area (mm²)'] / df['Lumen area (mm²)'].max()
df['Elliptic Ratio'] = df['Elliptic Ratio'] / df['Elliptic Ratio'].max()
df['Elliptic Ratio'] = (df['Elliptic Ratio'] -1) * -1
df['Vector Angle'] = df['Vector Angle'] / df['Vector Angle'].max()
df['Vector length'] = df['Vector length'] / df['Vector length'].max()
df['Vector'] = (df['Vector Angle'] + df['Vector length']) / 2
df['Vector'] = (df['Vector'] -1) * -1

# Apply a moving average to smooth the curves
window_size = 7  # Adjust the window size as needed
df['Smoothed Lumen area'] = df['Lumen area (mm²)'].rolling(window=window_size).mean()
df['Smoothed Elliptic Ratio'] = df['Elliptic Ratio'].rolling(window=window_size).mean()
df['Smoothed Vector'] = df['Vector'].rolling(window=window_size).mean()

combinations = []

# Loop through all possible values for the first element (0.01 to 0.99)
for a in range(1, 100):
    a_value = a * 0.01

    # Loop through all possible values for the second element (0.01 to remaining balance)
    for b in range(1, 100 - a):
        b_value = b * 0.01

        # The third element is the remaining balance
        c_value = 1 - a_value - b_value

        # Add the combination to the list if it sums to 1
        if c_value >= 0:
            combinations.append([round(a_value,2), round(b_value,2), round(c_value, 2)])

# Define a function to calculate the variance of local maxima differences
def variance_of_local_extrema_weights(weights, df, comparison_function):
    alpha, beta, gamma = weights
    df['Signal'] = alpha * df['Smoothed Lumen area'] + beta * df['Smoothed Elliptic Ratio'] + gamma * df['Smoothed Vector']

    local_extrema_indices = argrelextrema(df['Signal'].values, comparison_function, order=5)
    local_extrema_indices = local_extrema_indices[0].tolist()
    local_extrema_differences = [local_extrema_indices[i] - local_extrema_indices[i-1] for i in range(1, len(local_extrema_indices))]

    return np.var(local_extrema_differences)

variance_greater = []
for i in combinations:
    variance_greater.append(variance_of_local_extrema_weights(i, df, np.greater))

variance_lesser = []
for i in combinations:
    variance_lesser.append(variance_of_local_extrema_weights(i, df, np.less))

# print the result at the minimum index
weights_greater = (combinations[np.argmin(variance_greater)])
weights_lesser = (combinations[np.argmin(variance_lesser)])

variance_of_local_extrema_weights(weights_greater, df, np.greater)
variance_of_local_extrema_weights(weights_lesser, df, np.less)

# Find the indices of local minima in the 'Signal' column
local_minima_indices = argrelextrema(df['Signal'].values, np.less, order=5)
local_maxima_indices = argrelextrema(df['Signal'].values, np.greater, order=5)

# Convert the result to a list of indices
local_minima_indices = local_minima_indices[0].tolist()
local_maxima_indices = local_maxima_indices[0].tolist()
local_minima_differences = [local_minima_indices[i] - local_minima_indices[i-1] for i in range(1, len(local_minima_indices))]
local_maxima_differences = [local_maxima_indices[i] - local_maxima_indices[i-1] for i in range(1, len(local_maxima_indices))]
print(df['Frame'].iloc[local_maxima_indices])
print(df['Frame'].iloc[local_minima_indices])
print(np.var(local_minima_differences))
print(np.var(local_maxima_differences))


# Add a "D" to the 'Phase' column on the corresponding rows
df['Phase'] = '-'
df.loc[local_minima_indices, 'Phase'] = 'D'
df.loc[local_maxima_indices, 'Phase'] = 'S'

# Plot frame on x-axis and elliptic ratio and lumen area on y-axis
fig, ax = plt.subplots()
ax.plot(df['Frame'], df['Smoothed Elliptic Ratio'], label='Elliptic Ratio')
ax.plot(df['Frame'], df['Smoothed Lumen area'], label='Lumen area (mm²)')
ax.plot(df['Frame'], df['Signal'], label='Signal')
ax.plot(df['Frame'], df['Smoothed Vector'], label='Vector')
ax.set_xlabel('Frame')
ax.set_ylabel('Elliptic Ratio, Lumen area (mm²), and Signal')
ax.set_title('Elliptic Ratio, Lumen area (mm²), and Signal by Frame')
ax.legend()

# Scatter plot for 'S' (local maxima) and 'D' (local minima)
ax.scatter(df['Frame'].iloc[local_maxima_indices], df['Signal'].iloc[local_maxima_indices], color='red', marker='o', label='S')
ax.scatter(df['Frame'].iloc[local_minima_indices], df['Signal'].iloc[local_minima_indices], color='blue', marker='o', label='D')

plt.show()

# write results to csv file in path directory
df.to_csv('/home/yungselm/Documents/testreport_long_calc.csv', index=False)

print(local_minima_differences)
print(local_maxima_differences)