# plotting the elliptic ratio and area
# read text file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema

# Specify the file path
file_path = '/home/yungselm/Documents/IVUSimages/NARCO_119/20230809/125657/Run1/PDBHSCIO_report.txt'

# Read the data from the file into a DataFrame
df = pd.read_csv(file_path, delimiter='\t')

def data_preparation(df, window_size=13):
    # normalize lumen area and elliptic ratio
    df['Lumen area (mm²)'] = df['Lumen area (mm²)'] / df['Lumen area (mm²)'].max()
    df['Elliptic Ratio'] = df['Elliptic Ratio'] / df['Elliptic Ratio'].max()
    df['Elliptic Ratio'] = (df['Elliptic Ratio'] -1) * -1
    df['Vector Angle'] = df['Vector Angle'] / df['Vector Angle'].max()
    df['Vector length'] = df['Vector length'] / df['Vector length'].max()
    df['Vector'] = (df['Vector Angle'] + df['Vector length']) / 2
    df['Vector'] = (df['Vector'] -1) * -1

    # Apply a moving average to smooth the curves
    window_size = window_size  # Adjust the window size as needed
    df['Smoothed Lumen area'] = df['Lumen area (mm²)'].rolling(window=window_size).mean()
    df['Smoothed Elliptic Ratio'] = df['Elliptic Ratio'].rolling(window=window_size).mean()
    df['Smoothed Vector'] = df['Vector'].rolling(window=window_size).mean()
    return df


def lookup_table_weights_combinations(step=0.01):
    combinations = []

    # Determine the range based on the step size
    num_values = int(1 / step)

    # Loop through all possible values for the first element
    for a in range(1, num_values + 1):
        a_value = a * step

        # Loop through all possible values for the second element
        for b in range(1, num_values + 1 - a):
            b_value = b * step

            # Calculate the third element
            c_value = 1 - a_value - b_value

            # Add the combination to the list if it sums to 1
            if c_value >= 0:
                combinations.append([round(a_value, 2), round(b_value, 2), round(c_value, 2)])

    return combinations


# Define a function to calculate the variance of local maxima differences
def variance_of_local_extrema_weights(weights, df, comparison_function):
    alpha, beta, gamma = weights
    df['Signal'] = alpha * df['Smoothed Lumen area'] + beta * df['Smoothed Elliptic Ratio'] + gamma * df['Smoothed Vector']

    local_extrema_indices = argrelextrema(df['Signal'].values, comparison_function, order=5)
    local_extrema_indices = local_extrema_indices[0].tolist()
    local_extrema_differences = [local_extrema_indices[i] - local_extrema_indices[i-1] for i in range(1, len(local_extrema_indices))]

    return np.var(local_extrema_differences)


def optimize_window_size_and_weights(df, min_window_size, max_window_size, step=0.01):
    df_original = data_preparation(df)
    min_window_size = min_window_size
    max_window_size = max_window_size
    combinations = lookup_table_weights_combinations(step=step)

    variance_maxima = []
    variance_minima = []
    weights_greater = []
    weights_lesser = []
    window_sizes = []
    # add up every values at the same index

    for i in range(min_window_size, max_window_size + 1):
        print(f'{i - min_window_size} of {max_window_size - min_window_size + 1} iterations done')
        df = df_original.copy()
        df = data_preparation(df, window_size=i)
        variance_greater = []
        for j in combinations:
            variance_greater.append(variance_of_local_extrema_weights(j, df, np.greater))

        variance_lesser = []
        for j in combinations:
            variance_lesser.append(variance_of_local_extrema_weights(j, df, np.less))

        weights_max = combinations[np.argmin(variance_greater)]
        weights_min = combinations[np.argmin(variance_lesser)]

        variance_max = variance_of_local_extrema_weights(weights_max, df, np.greater)
        variance_min = variance_of_local_extrema_weights(weights_min, df, np.less)

        weights_greater.append(weights_max)
        weights_lesser.append(weights_min)
        variance_maxima.append(variance_max)
        variance_minima.append(variance_min)
        window_sizes.append(i)
    print(f'{max_window_size - min_window_size + 1} of {max_window_size - min_window_size + 1} iterations done')

    # find the minimum variance in the result and return the index
    sum_var = [sum(x) for x in zip(variance_maxima, variance_minima)]
    #find min index and use this to find value at same index in window_size
    print(sum_var)
    idx = np.argmin(sum_var)
    window = window_sizes[idx]
    weights_systole = weights_greater[idx]
    weigths_diastole = weights_lesser[idx]
    print(f'optimal window size is {window}')
    print(f'optimal weights for systole are {weights_systole}')
    print(f'optimal weights for diastole are {weigths_diastole}')
    # min_variance_index = np.argmin(result)  # add 1 to get the window size

    return weights_systole, weigths_diastole, window_sizes

weights_systole, weigths_diastole, window_sizes = optimize_window_size_and_weights(df, 2, 5)


def column_diastole_systole(df, combinations, variance_greater, variance_lesser):
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

    # Add a "D" to the 'Phase' column on the corresponding rows
    df['Phase (algorithm)'] = '-'
    df.loc[local_minima_indices, 'Phase (algorithm)'] = 'D'
    df.loc[local_maxima_indices, 'Phase (algorithm)'] = 'S'

    return df, local_minima_indices, local_maxima_indices, local_minima_differences, local_maxima_differences


def plot_results(df, local_minima_indices, local_maxima_indices):
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

    return plt

# df = data_preparation(df)

# combinations = lookup_table_weights_combinations(step=0.01)

# variance_greater = []
# for i in combinations:
#     variance_greater.append(variance_of_local_extrema_weights(i, df, np.greater))

# variance_lesser = []
# for i in combinations:
#     variance_lesser.append(variance_of_local_extrema_weights(i, df, np.less))

# df, local_minima_indices, local_maxima_indices, local_minima_differences, local_maxima_differences = column_diastole_systole(df, combinations, variance_greater, variance_lesser)

# plt = plot_results(df, local_minima_indices, local_maxima_indices)
# plt.show()

# print(df)
# print(local_minima_differences)
# print(local_maxima_differences)

# write results to csv file in path directory
df.to_csv('/home/yungselm/Documents/testreport_long_calc.csv', index=False)