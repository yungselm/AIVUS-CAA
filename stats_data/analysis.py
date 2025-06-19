import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

import pydicom as dcm

# Constants for pullback speed and frame rate
PULLBACK_SPEED = 1  
FRAME_RATE = 30    
PULLBACK_START_FRAME = 100 

def create_rest_signal(path) -> pd.DataFrame:
    df = pd.DataFrame(columns=['frame', 'signal_maxima', 'signal_extrema', 'signal_maxima_nor', 'signal_extrema_nor'])
    with open(path) as f:
        data = json.load(f)
        frames = len(data['gating_signal']['image_based_gating'])
        rows = []
        # for i in range(frames):
        #     rows.append({
        #         'frame': i + 1,
        #         'image_based': data['gating_signal']['image_based_gating'][i],
        #         'contour_based': data['gating_signal']['contour_based_gating'][i],
        #         'image_based_filtered': data['gating_signal']['image_based_gating_filtered'][i],
        #         'contour_based_gating': data['gating_signal']['contour_based_gating'][i]
        #     })
        for i in range(frames):
            rows.append({
                'frame': i + 1,
                'signal_maxima_nor': data['gating_signal']['image_based_gating'][i],
                'signal_extrema_nor': data['gating_signal']['contour_based_gating'][i],
                'signal_maxima': data['gating_signal']['image_based_gating_filtered'][i],
                'signal_extrema': data['gating_signal']['contour_based_gating_filtered'][i]
            })

        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)

        # all values /1000
        df['signal_maxima'] = df['signal_maxima'] / 1000
        df['signal_extrema'] = df['signal_extrema'] / 1000
        df['signal_maxima_nor'] = df['signal_maxima_nor'] / 1000
        df['signal_extrema_nor'] = df['signal_extrema_nor'] / 1000

        # flip 180 degrees around x-axis
        df['signal_maxima'] = df['signal_maxima'] * -1
        df['signal_extrema'] = df['signal_extrema'] * -1
        df['signal_maxima_nor'] = df['signal_maxima_nor'] * -1
        df['signal_extrema_nor'] = df['signal_extrema_nor'] * -1

        df.to_csv('output/flow_loop_rest_signals.csv')

        return df
    
# def create_icc_dataframe(folder_path):
#     frame_niva_run1 = []
#     phase_niva_run1 = []
#     frame_eye_run1 = []
#     phase_eye_run1 = []
#     frame_niva_run2 = []
#     phase_niva_run2 = []
#     frame_eye_run2 = []
#     phase_eye_run2 = []

#     for file in os.listdir(folder_path):
#         # open every .txt file
#         with open(folder_path + file) as f:
#             data = pd.read_csv(f, sep='\t')
#             data = data[data['phase'] != '-'][['frame', 'phase']]
#             # if file contains 'niva' in name and does not contain run2
#             if 'niva' in file and 'run2' not in file:
#                 print("niva run1 has length", len(data))
#                 frames = data['frame'].tolist()
#                 phases = data['phase'].tolist()
#                 frame_niva_run1.extend(frames)
#                 phase_niva_run1.extend(phases)
#             # if file contains 'eye' in name and does not contain run2
#                 frames = data['frame'].tolist()
#                 phases = data['phase'].tolist()
#                 frame_eye_run1.extend(frames)
#                 phase_eye_run1.extend(phases)
#                 phase_eye_run1.append(frames[1])
#                 frames = data['frame'].tolist()
#                 phases = data['phase'].tolist()
#                 frame_niva_run2.extend(frames)
#                 phase_niva_run2.extend(phases)
#                 frame_niva_run2.append(frames[0])
#                 frames = data['frame'].tolist()
#                 phases = data['phase'].tolist()
#                 frame_eye_run2.extend(frames)
#                 phase_eye_run2.extend(phases)
#                 frames = list(data.columns.values)
#                 frame_eye_run2.append(frames[0])
#                 phase_eye_run2.append(frames[1])

        
#     # create one dataframe with each list as column
#     df = pd.DataFrame({
#         'frame_niva_run1': frame_niva_run1,
#         'phase_niva_run1': phase_niva_run1,
#         'frame_eye_run1': frame_eye_run1,
#         'phase_eye_run1': phase_eye_run1,
#         'frame_niva_run2': frame_niva_run2,
#         'phase_niva_run2': phase_niva_run2,
#         'frame_eye_run2': frame_eye_run2,
#         'phase_eye_run2': phase_eye_run2
#     })

#     df.to_csv('output/icc.csv')
#     print(df)

import os
import pandas as pd

def create_icc_dataframe(folder_path):
    # Initialize lists for each column
    frame_niva_run1 = []
    phase_niva_run1 = []
    frame_eye_run1 = []
    phase_eye_run1 = []
    frame_niva_run2 = []
    phase_niva_run2 = []
    frame_eye_run2 = []
    phase_eye_run2 = []

    # Iterate through all files in the folder
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        
        # Process only text files
        if file.endswith('.txt'):
            with open(file_path) as f:
                data = pd.read_csv(f, sep='\t')
                data = data[data['phase'] != '-'][['frame', 'phase']]  # Filter and select relevant columns
                
                # Populate lists based on file name conditions
                if 'niva' in file and 'run2' not in file:
                    print(file, "has length", len(data))
                    frame_niva_run1.extend(data['frame'].tolist())
                    phase_niva_run1.extend(data['phase'].tolist())
                elif 'eye' in file and 'run2' not in file:
                    print(file, "has length", len(data))
                    frame_eye_run1.extend(data['frame'].tolist())
                    phase_eye_run1.extend(data['phase'].tolist())
                elif 'niva' in file and 'run2' in file:
                    print(file, "has length", len(data))
                    frame_niva_run2.extend(data['frame'].tolist())
                    phase_niva_run2.extend(data['phase'].tolist())
                elif 'eye' in file and 'run2' in file:
                    print(file, "has length", len(data))
                    frame_eye_run2.extend(data['frame'].tolist())
                    phase_eye_run2.extend(data['phase'].tolist())

    # Ensure all lists are the same length by padding with None
    max_length = max(
        len(frame_niva_run1), len(phase_niva_run1),
        len(frame_eye_run1), len(phase_eye_run1),
        len(frame_niva_run2), len(phase_niva_run2),
        len(frame_eye_run2), len(phase_eye_run2)
    )

    # Pad lists with None
    def pad_list(lst, target_length):
        return lst + [None] * (target_length - len(lst))
    
    frame_niva_run1 = pad_list(frame_niva_run1, max_length)
    phase_niva_run1 = pad_list(phase_niva_run1, max_length)
    frame_eye_run1 = pad_list(frame_eye_run1, max_length)
    phase_eye_run1 = pad_list(phase_eye_run1, max_length)
    frame_niva_run2 = pad_list(frame_niva_run2, max_length)
    phase_niva_run2 = pad_list(phase_niva_run2, max_length)
    frame_eye_run2 = pad_list(frame_eye_run2, max_length)
    phase_eye_run2 = pad_list(phase_eye_run2, max_length)

    # Create the DataFrame
    df = pd.DataFrame({
        'frame_niva_run1': frame_niva_run1,
        'phase_niva_run1': phase_niva_run1,
        'frame_eye_run1': frame_eye_run1,
        'phase_eye_run1': phase_eye_run1,
        'frame_niva_run2': frame_niva_run2,
        'phase_niva_run2': phase_niva_run2,
        'frame_eye_run2': frame_eye_run2,
        'phase_eye_run2': phase_eye_run2
    })

    # Save to CSV
    output_path = os.path.join('output', 'icc.csv')
    os.makedirs('output', exist_ok=True)  # Ensure the output directory exists
    df.to_csv(output_path, index=False)
    print(df)

# Call the function
create_icc_dataframe('icc/')


# flow_loop_rest = dcm.dcmread('input/flow_loop_rest')
# fl_rest_data = pd.read_csv('input/PDD6U1UJ_report.txt', sep='\t')
# fl_rest_signal = create_rest_signal('input/PDD6U1UJ_contours_0_7_4.json')