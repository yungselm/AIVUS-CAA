import itertools
import numpy as np
from loguru import logger

from gating.signal_processing import identify_extrema
from gui.popup_windows.frame_range_dialog import StartFramesDialog


class AutomaticGating:
    def __init__(self, main_window, report_data) -> None:
        self.main_window = main_window
        self.report_data = report_data
        self.maxima_only = self.main_window.config.gating.maxima_only
        self.auto_gating_threshold = main_window.config.gating.auto_gating_threshold
        self.x = self.report_data['frame'].values  # want 1-based indexing for GUI

    def automatic_gating(self, image_based_signal, contour_based_signal):
        """
        Automatically gates the frames based on the maxima and extrema signals.
        The maxima signal represents the image-based gating and the extrema signal
        the contour-based gating. Usually maxima should be overlapping with extrema, but
        this is not always the case. Therefore the option to change in the config.yaml file
        The gating is based on the following assumptions:
        - Diastole frames can depict more distal parts of the coronary artery, and AAOCA undergoes
            more compression during systole, hence sum of lumen area is higher for diastolic frames.

        Parameters:
        - image_based_signal (numpy.ndarray): The signal containing maxima.
        - contour_based_signal (numpy.ndarray): The signal containing extrema.
        """
        dialog = StartFramesDialog(self.main_window)
        if dialog.exec_():
            if self.maxima_only:
                logger.info('Maxima peak detection according to config.yaml')
                maxima_indices = identify_extrema(self.main_window, image_based_signal)[1]
                extrema_indices = list(identify_extrema(self.main_window, contour_based_signal)[0])
            else:
                logger.info('Extrema peak detection according to config.yaml')
                maxima_indices = identify_extrema(self.main_window, image_based_signal)[0]
                extrema_indices = list(identify_extrema(self.main_window, contour_based_signal)[0])

            heart_rate = (
                int(self.estimate_frame_distance(maxima_indices) + self.estimate_frame_distance(extrema_indices)) / 2
            )

            # Initialize lists for diastole and systole frames
            similarity_dia = []
            similarity_sys = []

            systolic_start, diastolic_start = None, None
            try:
                # Get the start frames for diastole and systole
                systolic_start, diastolic_start = dialog.getInputs()
            except:
                if systolic_start is None or diastolic_start is None:
                    # if systolic_start or diastolic_start is not defined, find the first and second maxima indices and extrema indices that are not further then 5 frames apart
                    # starting from the last index in both lists
                    found_first = False  # To track the first occurrence for diastole
                    # Loop from the end of maxima_indices and extrema_indices to find close pairs
                    for max_idx in reversed(maxima_indices):
                        for ext_idx in reversed(extrema_indices):
                            # Check if indices are within 5 frames of each other
                            if abs(max_idx - ext_idx) <= 5:
                                # Set diastolic_start to the first occurrence, and systolic_start to the second
                                if not found_first:
                                    diastolic_start = (max_idx + ext_idx) // 2
                                    found_first = True  # Mark that we've found the first occurrence
                                else:
                                    systolic_start = (max_idx + ext_idx) // 2
                                    break  # Exit once both are found
                        if systolic_start is not None and diastolic_start is not None:
                            break

                # Log the selected start frames
                logger.info(f"Systolic start frame: {systolic_start}, Diastolic start frame: {diastolic_start}")

            # Find and append diastolic frames based on correlation
            current_frame = diastolic_start
            combined_signal_dia = []
            combined_signal_sys = []

            while current_frame is not None:
                similarity_dia.append(current_frame)
                correlations, frame_indices = self.correlation_automatic(current_frame, heart_rate)
                current_frame, _, indices_dia, signal_dia = self.find_best_correlation(
                    current_frame, correlations, frame_indices
                )
                combined_signal_dia.append(signal_dia)

            # Find and append systolic frames based on correlation
            current_frame = systolic_start
            while current_frame is not None:
                similarity_sys.append(current_frame)
                correlations, frame_indices = self.correlation_automatic(current_frame, heart_rate)
                current_frame, _, indices_sys, signal_sys = self.find_best_correlation(
                    current_frame, correlations, frame_indices
                )
                combined_signal_sys.append(signal_sys)

            combined_signal_dia = np.concatenate(combined_signal_dia)
            combined_signal_sys = np.concatenate(combined_signal_sys)

            # add 0 until length of extrema_indices
            while len(combined_signal_dia) < len(self.x):
                combined_signal_dia = np.append(combined_signal_dia, 0)
            while len(combined_signal_sys) < len(self.x):
                combined_signal_sys = np.append(combined_signal_sys, 0)

            # in combined_signal_dia and combined_signal_sys, replace 0 with the minimal value of the signal
            combined_signal_dia[combined_signal_dia == 0] = np.min(combined_signal_dia[combined_signal_dia != 0])
            combined_signal_sys[combined_signal_sys == 0] = np.min(combined_signal_sys[combined_signal_sys != 0])

            similarity_indices = similarity_dia + similarity_sys

            gated_indices = []

            # Calculate weights for maxima and extrema based on variability or signal quality
            maxima_weight = len(maxima_indices)
            extrema_weight = len(extrema_indices)

            # Check for all indices in maxima and extrema if they are not more than 5 frames apart
            for maxima in maxima_indices:
                # Find up to three closest extrema within the specified range
                close_extrema = [
                    extrema for extrema in extrema_indices if abs(maxima - extrema) <= self.auto_gating_threshold
                ]

                # If there are enough close extrema, proceed with variability check
                if len(close_extrema) >= 3:
                    # Generate combinations of three extrema and calculate variability
                    variability_scores = []
                    extrema_combinations = itertools.combinations(close_extrema, 3)

                    for combo in extrema_combinations:
                        # Calculate variability as the standard deviation of lumen areas for this combination
                        lumen_areas = [
                            self.report_data.loc[self.report_data['frame'] == frame, 'lumen_area'].values[0]
                            for frame in combo
                        ]
                        variability = np.std(lumen_areas)
                        variability_scores.append((combo, variability))

                    # Select the combination with the least variability
                    best_combo = min(variability_scores, key=lambda x: x[1])[0]

                    # Use the middle frame of the best combination as the gated frame
                    closest_extrema = sorted(best_combo)[1]
                    gated_indices.append(round((maxima + closest_extrema) / 2))

                    # Remove the selected extrema from further consideration
                    extrema_indices = [e for e in extrema_indices if e not in best_combo]

                elif close_extrema:
                    # If fewer than three close extrema, select the closest one
                    closest_extrema = close_extrema[0]
                    gated_indices.append(round((maxima + closest_extrema) / 2))
                    extrema_indices.remove(closest_extrema)  # Remove to avoid duplicate gating

            # Split by every second frame; diastole frames are where lumen_area is greater in sum than the other half
            first_half = gated_indices[::2]
            second_half = gated_indices[1::2]

            sum_first_half = sum(
                [
                    self.report_data.loc[self.report_data['frame'] == frame, 'lumen_area'].values[0]
                    for frame in first_half
                ]
            )
            sum_second_half = sum(
                [
                    self.report_data.loc[self.report_data['frame'] == frame, 'lumen_area'].values[0]
                    for frame in second_half
                ]
            )

            # reset all phases
            self.main_window.data['phases'] == '-'
            self.main_window.gated_frames_dia = []
            self.main_window.gated_frames_sys = []
            self.main_window.diastolic_frame_box.setChecked(False)
            self.main_window.systolic_frame_box.setChecked(False)

            if sum_first_half > sum_second_half:
                self.main_window.gated_frames_dia = first_half
                self.main_window.gated_frames_sys = second_half
                self.main_window.gated_frames_dia.sort()
                self.main_window.gated_frames_sys.sort()
            else:
                self.main_window.gated_frames_dia = second_half
                self.main_window.gated_frames_sys = first_half
                self.main_window.gated_frames_dia.sort()
                self.main_window.gated_frames_sys.sort()

            for frame in self.main_window.gated_frames_dia:
                self.main_window.data['phases'][frame] = 'D'
            for frame in self.main_window.gated_frames_sys:
                self.main_window.data['phases'][frame] = 'S'

    def estimate_frame_distance(self, indices):
        """Estimates the average frame distance based on the gated frames using every second interval.
        input: vector of indices of gated frames"""
        # Ensure there are enough indices to calculate the frame distance
        if len(indices) < 3:
            print("Not enough indices to calculate frame distance.")
            return None

        frame_distances = []

        # Loop through indices with a step of 2
        for i in range(2, len(indices), 2):
            # Calculate frame interval between every second entry
            frame_diff = indices[i] - indices[i - 2]
            if frame_diff > 0:
                frame_distances.append(frame_diff)
            else:
                print("Warning: Zero frame difference encountered.")

        # Calculate the mean frame distance if there are valid entries
        if frame_distances:
            mean_frame_distance = np.mean(frame_distances)
            mean_frame_distance = int(mean_frame_distance)
            print("Estimated Frame Distance:", mean_frame_distance)
            return mean_frame_distance
        else:
            print("Unable to calculate frame distance from given indices.")
            return None

    def correlation_automatic(self, frame, heart_rate):
        """Calculates correlation coefficients with the previous 20 to 10 frames."""
        correlations = []
        frame_indices = []

        if heart_rate >= 12:  # corresponding to 100 bpm
            start_frame = max(0, frame - int(heart_rate + 10))
            end_frame = max(0, frame - int(heart_rate - 10))
        else:
            start_frame = max(0, frame - int(heart_rate + 5))
            end_frame = max(0, frame - int(heart_rate - 5))

        for i in range(start_frame, end_frame):
            corr = np.corrcoef(self.main_window.images[frame].ravel(), self.main_window.images[i].ravel())[0, 1]
            correlations.append(corr)
            frame_indices.append(i)

        # If less than 10 frames, pad with 0s to maintain the length
        if heart_rate >= 12:
            while len(correlations) < 10:
                correlations.insert(0, 0)  # Prepend zeros if necessary
                frame_indices.insert(0, None)  # Prepend None for frame indices
        else:
            while len(correlations) < 5:
                correlations.insert(0, 0)
                frame_indices.insert(0, None)

        return correlations, frame_indices

    def find_best_correlation(self, frame, correlations, frame_indices):
        """Finds the frame with the highest correlation."""
        if not correlations or not frame_indices:
            return None, None, [], []  # Return empty lists if correlations or indices are not available

        max_corr = max(correlations)
        max_index = correlations.index(max_corr)

        if max_index < 0 or max_index >= len(frame_indices):
            return None, None, [], []  # Guard against invalid index access

        best_frame_index = frame_indices[max_index]

        # Initialize lists to build the new signal and indices
        new_indices = []
        new_correlations = []

        # Iterate from the best frame index to the specified frame
        if best_frame_index is not None:
            for idx in range(best_frame_index, frame):
                if idx in frame_indices:
                    # Find the index in frame_indices to get the correlation
                    original_index = frame_indices.index(idx)
                    new_indices.append(idx)
                    new_correlations.append(correlations[original_index])
                else:
                    # For any indices not in frame_indices, we set correlation to 0
                    new_indices.append(idx)
                    new_correlations.append(0)
        else:
            # If no best frame index found, return empty lists
            return None, None, [], []

        return best_frame_index, max_corr, new_indices, new_correlations
