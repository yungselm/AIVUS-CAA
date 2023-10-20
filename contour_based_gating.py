# plotting the elliptic ratio and area
# read text file
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import argrelextrema
from tqdm import tqdm

# Specify the file path
file_path = '/home/yungselm/Documents/IVUSimages/NARCO_119/20230809/125657/Run1/PDBHSCIO_report.txt' # rest
# file_path = '/home/yungselm/Documents/IVUSimages/NARCO_119/20230809/125657/Run3/PDOXMQUN_report.txt' # stress

# Read the data from the file into a DataFrame
df = pd.read_csv(file_path, delimiter='\t')
# split the data into two dataframes based on elliptic ratio with cut_off 1.3
df_im = df[df['Elliptic Ratio'] > 1.3]
df_not_im = df[df['Elliptic Ratio'] < 1.3]

def data_preparation(df, window_size=13):
    df_copy = df.copy()
    # Normalize lumen area and elliptic ratio
    df_copy['Lumen area (mm²)'] = df_copy['Lumen area (mm²)'] / df_copy['Lumen area (mm²)'].max()
    df_copy['Elliptic Ratio'] = df_copy['Elliptic Ratio'] / df_copy['Elliptic Ratio'].max()
    df_copy['Elliptic Ratio'] = (df_copy['Elliptic Ratio'] - 1) * -1
    df_copy['Vector Angle'] = df_copy['Vector Angle'] / df_copy['Vector Angle'].max()
    df_copy['Vector length'] = df_copy['Vector length'] / df_copy['Vector length'].max()
    df_copy['Vector'] = (df_copy['Vector Angle'] + df_copy['Vector length']) / 2
    df_copy['Vector'] = (df_copy['Vector'] - 1) * -1

    # Apply a moving average to smooth the curves
    window_size = window_size  # Adjust the window size as needed
    df_copy['Smoothed Lumen area'] = df_copy['Lumen area (mm²)'].rolling(window=window_size).mean()
    df_copy['Smoothed Elliptic Ratio'] = df_copy['Elliptic Ratio'].rolling(window=window_size).mean()
    df_copy['Smoothed Vector'] = df_copy['Vector'].rolling(window=window_size).mean()

    return df_copy


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

    local_extrema = argrelextrema(df['Signal'].values, comparison_function, order=5)
    local_extrema = local_extrema[0].tolist()
    local_extrema_indices = df.index[local_extrema].tolist()
    local_extrema_differences = [local_extrema_indices[i] - local_extrema_indices[i-1] for i in range(1, len(local_extrema_indices))]
    return local_extrema_differences, local_extrema_indices, df['Signal']


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
    systolic_idx = []
    diastolic_idx = []
    signals_systole = []
    signals_diastole = []

    for i in tqdm(range(min_window_size, max_window_size + 1)):
        df_loop = df_original.copy()
        df_loop = data_preparation(df_loop, window_size=i)
        variance_greater = []
        local_maxima_indices = []
        signals_sys = []
        for j in combinations:
            local_extrema_differences, local_extrema_indices, signal = variance_of_local_extrema_weights(j, df_loop, np.greater)
            variance_greater.append(np.var(local_extrema_differences))
            local_maxima_indices.append(local_extrema_indices)
            signals_sys.append(signal)

        variance_lesser = []
        local_minima_indices = []
        signals_dia = []
        for j in combinations:
            local_extrema_differences, local_extrema_indices, signal = variance_of_local_extrema_weights(j, df_loop, np.less)
            variance_lesser.append(np.var(local_extrema_differences))
            local_minima_indices.append(local_extrema_indices)
            signals_dia.append(signal)

        weights_max = combinations[np.argmin(variance_greater)]
        weights_min = combinations[np.argmin(variance_lesser)]

        variance_max = np.min(variance_greater)
        variance_min = np.min(variance_lesser)

        systolic_indices_loop = local_maxima_indices[np.argmin(variance_greater)]
        diastolic_indices_loop = local_minima_indices[np.argmin(variance_lesser)]

        signals_sys_min = signals_sys[np.argmin(variance_greater)]
        signals_dia_min = signals_dia[np.argmin(variance_lesser)]

        variance_maxima.append(variance_max)
        variance_minima.append(variance_min)
        weights_greater.append(weights_max)
        weights_lesser.append(weights_min)
        window_sizes.append(i)
        systolic_idx.append(systolic_indices_loop)
        diastolic_idx.append(diastolic_indices_loop)
        signals_systole.append(signals_sys_min)
        signals_diastole.append(signals_dia_min)

    # find the minimum variance in the result and return the index
    sum_var = [sum(x) for x in zip(variance_maxima, variance_minima)]

    #find min index and use this to find value at same index in window_size
    idx = np.argmin(sum_var)
    window = window_sizes[idx]
    weights_systole = weights_greater[idx]
    weigths_diastole = weights_lesser[idx]
    systolic_indices = systolic_idx[idx]
    diastolic_indices = diastolic_idx[idx]
    signal_systole = signals_systole[idx]
    signal_diastole = signals_diastole[idx]
    print(f'optimal window size: {window}')
    print(f'\033[91mOptimal weights Systole:\033[0m\nLumen area: {weights_systole[0]}, Elliptic Ratio: {weights_systole[1]}, Vector: {weights_systole[2]}')
    print(f'\033[94mOptimal weights Diastole:\033[0m\nLumen area: {weigths_diastole[0]}, Elliptic Ratio: {weigths_diastole[1]}, Vector: {weigths_diastole[2]}')

    # min_variance_index = np.argmin(result)  # add 1 to get the window size

    return weights_systole, weigths_diastole, window_sizes, systolic_indices, diastolic_indices, signal_systole, signal_diastole


def column_phase(df, systolic_indices, diastolic_indices, signal_systole, signal_diastole):
    systole = pd.Series(signal_systole)
    diastole = pd.Series(signal_diastole)
    df['Signal Systole'] = systole
    df['Signal Diastole'] = diastole
    df['Phase (algorithm)'] = '-'
    df.loc[diastolic_indices, 'Phase (algorithm)'] = 'D'
    df.loc[systolic_indices, 'Phase (algorithm)'] = 'S'

    return df


def plot_results(df, diastolic_indices, systolic_indices):
    # Plot frame on x-axis and elliptic ratio and lumen area on y-axis
    fig, ax = plt.subplots()
    ax.plot(df['Frame'], df['Smoothed Elliptic Ratio'], label='Elliptic Ratio')
    ax.plot(df['Frame'], df['Smoothed Lumen area'], label='Lumen area (mm²)')
    ax.plot(df['Frame'], df['Signal Systole'], label='Signal Systole')
    ax.plot(df['Frame'], df['Signal Diastole'], label='Signal Diastole')
    ax.plot(df['Frame'], df['Smoothed Vector'], label='Vector')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Elliptic Ratio, Lumen area (mm²), and Signal')
    ax.set_title('Elliptic Ratio, Lumen area (mm²), and Signal by Frame')
    ax.legend()

    #find frames corresponding to row in systolic_indices and diastolic_indices
    frames_systole = df.loc[systolic_indices, 'Frame'].tolist()
    frames_diastole = df.loc[diastolic_indices, 'Frame'].tolist()
    signal_systole = df.loc[systolic_indices, 'Signal Systole'].tolist()
    signal_diastole = df.loc[diastolic_indices, 'Signal Diastole'].tolist()

    # Scatter plot for 'S' (local maxima) and 'D' (local minima)
    ax.scatter(frames_systole, signal_systole, color='red', marker='o', label='S')
    ax.scatter(frames_diastole, signal_diastole, color='blue', marker='o', label='D')

    return plt

# weights_systole, weigths_diastole, window_sizes, systolic_indices, diastolic_indices, signal_systole, signal_diastole = optimize_window_size_and_weights(df, 2, 20)
weights_systole, weights_diastole, window_sizes, systolic_indices, diastolic_indices, signal_systole, signal_diastole = optimize_window_size_and_weights(df, 3, 3)
df = data_preparation(df, window_size=14)
df = column_phase(df, systolic_indices, diastolic_indices, signal_systole, signal_diastole)

plt = plot_results(df, diastolic_indices, systolic_indices)
plt.show()

# write results to csv file in path directory
df.to_csv('/home/yungselm/Documents/testreport_long_calc.csv', index=False)