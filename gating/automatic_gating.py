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
        self.batch_size = main_window.config.gating.auto_gating_batch_size
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
            propagated_indices_dia = self.propagate_gated_frames(
                diastolic_start, heart_rate, maxima_indices, extrema_indices
            )
            propagated_indices_sys = self.propagate_gated_frames(
                systolic_start, heart_rate, maxima_indices, extrema_indices
            )

            propagated_indices = np.sort(propagated_indices_dia + propagated_indices_sys)
            weight_maxima = self.weight_signal(maxima_indices[::2])
            weight_extrema = self.weight_signal(extrema_indices[::2])
            weight_propagated = self.weight_signal(propagated_indices[::2])

            # take shortest list shorten other two lists to this length
            min_length = min(len(maxima_indices), len(extrema_indices), len(propagated_indices))
            maxima_indices = maxima_indices[len(maxima_indices) - min_length :]
            extrema_indices = extrema_indices[len(extrema_indices) - min_length :]
            propagated_indices = propagated_indices[len(propagated_indices) - min_length :]

            # Convert lists to numpy arrays for weighted sum calculation
            maxima_indices = np.array(maxima_indices)
            extrema_indices = np.array(extrema_indices)
            propagated_indices = np.array(propagated_indices)

            # Use weighted sum for each index to determine the final gated frames
            weighted_sum = (
                weight_maxima * maxima_indices
                + weight_extrema * extrema_indices
                + weight_propagated * propagated_indices
            )
            gated_indices = list(
                np.round(weighted_sum / (weight_maxima + weight_extrema + weight_propagated)).astype(int)
            )

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

    def propagate_gated_frames(self, starting_frame, heart_rate, maxima_indices, extrema_indices):
        propagated_indices = []
        counter = 0
        frame = starting_frame
        while frame is not None:
            propagated_indices.append(frame)
            correlations, frame_indices = self.correlation_automatic(frame, heart_rate)
            frame, _, _, _ = self.find_best_correlation(frame, correlations, frame_indices)
            if counter == self.batch_size:
                frame = min(
                    min(extrema_indices, key=lambda x: abs(x - frame)),
                    min(maxima_indices, key=lambda x: abs(x - frame)),
                    key=lambda x: abs(x - frame),
                )  # Reset after each batch
                counter = 0
            counter += 1

        return propagated_indices

    def weight_signal(self, indices):
        pairwise_distances = np.diff(indices)
        return 1 - np.var(pairwise_distances)

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
