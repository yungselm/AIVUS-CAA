import pandas as pd
import matplotlib.pyplot as plt
import os

# Read in the data
# Specify the file path
file_path_rest = '/home/yungselm/Documents/CaseReport/rest.txt'
file_path_stress = '/home/yungselm/Documents/CaseReport/stress.txt'
file_path_out = '/home/yungselm/Documents/CaseReport'

def get_df_name(df):
    name =[x for x in globals() if globals()[x] is df][0]
    return name

def calculate_position(df):
    df = df.copy()  # Make a copy of the DataFrame to avoid SettingWithCopyWarning

    # Initialize a list to store the 'Position' values
    positions = [0]

    # Calculate 'Position' values for the DataFrame
    for i in range(1, len(df)):
        frame_diff = df['Frame'].iloc[i] - df['Frame'].iloc[i - 1]
        position = positions[-1] + (frame_diff * 1/30)
        positions.append(position)

    # From every index of position subtract the max value of position and return the positive
    positions = [(i - max(positions)) * -1 for i in positions]

    # Add the 'Position' column to the DataFrame
    df['Position'] = positions

    # Return the modified DataFrame
    return df

def plot_lines(df1, df2, path=None, xlim = None, circumference=False):
    df1_name = get_df_name(df1)
    df2_name = get_df_name(df2)

    if path is None:
        write_path = os.path.join(os.getcwd(), f'{df1_name}_vs_{df2_name}.png')
    else:
        write_path = os.path.join(path, f'{df1_name}_vs_{df2_name}.png')

    #final plot
    fig, ax = plt.subplots()
    ax.plot(df1['Position'], df1['Elliptic Ratio'], label=f'Elliptic Ratio ({df1_name})', linestyle=':', color='b')
    ax.plot(df1['Position'], df1['Lumen area (mm²)'], label=f'Lumen area (mm²) ({df1_name})', linestyle='-', color='b')
    if circumference==True:
        ax.plot(df1['Position'], df1['Circumf (mm)'], label=f'Circumference (mm) ({df1_name})', linestyle='-.', color='b')
    ax.plot(df2['Position'], df2['Elliptic Ratio'], label=f'Elliptic Ratio ({df2_name})', linestyle=':', color='r')
    ax.plot(df2['Position'], df2['Lumen area (mm²)'], label=f'Lumen area (mm²) ({df2_name})', linestyle='-', color='r')
    if circumference==True:
        ax.plot(df2['Position'], df2['Circumf (mm)'], label=f'Circumference (mm) ({df2_name})', linestyle='-.', color='r')
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel('Elliptic Ratio, Lumen area (mm²)')
    ax.set_title('Elliptic Ratio and Lumen area (mm²) by Position (mm)')
    ax.legend()
    if xlim is not None:
        ax.set_xlim(0, xlim)
    plt.show()
    
    #save plot in path
    fig.set_size_inches(10, 5)
    fig.savefig(write_path, dpi=300) 
    return plt

# Read the data from the file into a DataFrame
rest = pd.read_csv(file_path_rest, delimiter='\t')
stress = pd.read_csv(file_path_stress, delimiter='\t')

# Create a new data set for Phase D and S
# Phase D
rest_D = rest[rest['Phase'] == 'D']
stress_D = stress[stress['Phase'] == 'D']
rest_S = rest[rest['Phase'] == 'S']
stress_S = stress[stress['Phase'] == 'S']

# Call the function on the DataFrames and update them
rest_D = calculate_position(rest_D)
stress_D = calculate_position(stress_D)
rest_S = calculate_position(rest_S)
stress_S = calculate_position(stress_S)

#combine rest_D and rest_S into rest again
rest = pd.concat([rest_D, rest_S])
stress = pd.concat([stress_D, stress_S])
# order by descending position
rest = rest.sort_values(by=['Position'], ascending=False)
stress = stress.sort_values(by=['Position'], ascending=False)

plot_lines(rest_D, rest_S, path=file_path_out, xlim=10, circumference=True)
plot_lines(stress_D, stress_S, path=file_path_out, xlim=10, circumference=True)
plot_lines(rest_D, stress_D, path=file_path_out, xlim=10, circumference=True)
plot_lines(rest_S, stress_S, path=file_path_out, xlim=10, circumference=True)
plot_lines(rest, stress, path=file_path_out, xlim=10, circumference=True)

# Calculate the rolling mean with a window size of your choice (e.g., window=5) for numeric columns
window = 4
numeric_columns = ['Elliptic Ratio', 'Lumen area (mm²)', 'Circumf (mm)', 'Position']
rest_smoothed = rest[numeric_columns].rolling(window).mean()
stress_smoothed = stress[numeric_columns].rolling(window).mean()

plot_lines(rest_smoothed, stress_smoothed, path=file_path_out, xlim=10, circumference=True)