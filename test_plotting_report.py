# plotting the elliptic ratio and area
# read text file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema

# Specify the file path
file_path = '/home/yungselm/Documents/testreport_long.txt'

# Read the data from the file into a DataFrame
df = pd.read_csv(file_path, delimiter='\t')

# normalize lumen area and elliptic ratio
df['Lumen area (mm²)'] = df['Lumen area (mm²)'] / df['Lumen area (mm²)'].max()
df['Elliptic Ratio'] = df['Elliptic Ratio'] / df['Elliptic Ratio'].max()
df['Elliptic Ratio'] = (df['Elliptic Ratio'] -1) * -1

# Apply a moving average to smooth the curves
window_size = 6  # Adjust the window size as needed
df['Smoothed Lumen area'] = df['Lumen area (mm²)'].rolling(window=window_size).mean()
df['Smoothed Elliptic Ratio'] = df['Elliptic Ratio'].rolling(window=window_size).mean()

alpha = np.linspace(1, 0, len(df))

# Initialize the "Signal" column with NaN values
df['Signal'] = np.nan

# Iterate through alpha values and calculate the "Signal" for each alpha
for i in range(len(alpha)):
    df.loc[i, 'Signal'] = alpha[i] * df['Smoothed Lumen area'].iloc[i] + (1 - alpha[i]) * df['Smoothed Elliptic Ratio'].iloc[i]

# Find the indices of local minima in the 'Signal' column
local_minima_indices = argrelextrema(df['Signal'].values, np.less, order=5)
local_maxima_indices = argrelextrema(df['Signal'].values, np.greater, order=5)

# Convert the result to a list of indices
local_minima_indices = local_minima_indices[0].tolist()
local_maxima_indices = local_maxima_indices[0].tolist()
print(df['Frame'].iloc[local_maxima_indices])
print(df['Frame'].iloc[local_minima_indices])


# Add a "D" to the 'Phase' column on the corresponding rows
df['Phase'] = '-'
df.loc[local_minima_indices, 'Phase'] = 'D'
df.loc[local_maxima_indices, 'Phase'] = 'S'

# Plot frame on x-axis and elliptic ratio and lumen area on y-axis
fig, ax = plt.subplots()
ax.plot(df['Frame'], df['Smoothed Elliptic Ratio'], label='Elliptic Ratio')
ax.plot(df['Frame'], df['Smoothed Lumen area'], label='Lumen area (mm²)')
ax.plot(df['Frame'], df['Signal'], label='Signal')
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

# calculate the mean difference between i and i+1 in the local maxima and minima
distances_maxima = []
for i in range(len(local_maxima_indices)):
    if i < len(local_maxima_indices) - 1:
        distances_maxima.append(local_maxima_indices[i+1] - local_maxima_indices[i])
    else:
        break

distances_minima = []
for i in range(len(local_minima_indices)):
    if i < len(local_minima_indices) - 1:
        distances_maxima.append(local_minima_indices[i+1] - local_minima_indices[i])
    else:
        break

# print mean of the list
print(distances_minima)
print(np.mean(distances_minima))
print(np.mean(distances_maxima))
