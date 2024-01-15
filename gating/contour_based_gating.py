import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import Qt
from scipy.signal import argrelextrema

from gui.error_message import ErrorMessage
from gui.frame_range_dialog import FrameRangeDialog
from report.report import report


class ContourBasedGating:
    def __init__(self, main_window):
        self.main_window = main_window
        self.intramural_threshold = main_window.config.gating.intramural_threshold
        self.min_window_size = main_window.config.gating.min_window_size
        self.max_window_size = main_window.config.gating.max_window_size
        self.step_size = main_window.config.gating.step_size

    def __call__(self):
        self.main_window.status_bar.showMessage('Contour-based gating...')
        self.report_data = report(self.main_window, suppress_messages=True)  # compute all needed data
        if self.report_data is None:
            ErrorMessage(self.main_window, 'Please ensure that a DICOM file was read and contours were drawn')
            self.main_window.status_bar.showMessage(self.main_window.waiting_status)
            return

        self.define_intramural_part()
        self.data_preparation()
        success = self.optimize_window_size_and_weights()
        if success:
            self.propagate_gating()
            self.update_main_window()
            self.plot_results()

        self.main_window.status_bar.showMessage(self.main_window.waiting_status)

    def define_intramural_part(self):
        dialog = FrameRangeDialog(self.main_window)
        if dialog.exec_():
            lower_limit, upper_limit = dialog.getInputs()
            if (
                lower_limit == 0 and upper_limit == self.main_window.images.shape[0]
            ):  # automatic detection of intramural part
                mean_elliptic_ratio = self.report_data['elliptic_ratio'].rolling(window=5, closed='both').mean()

            self.report_data = self.report_data.iloc[lower_limit:upper_limit, :]

    def data_preparation(self):
        self.report_data['lumen_area'] = self.report_data['lumen_area'] / self.report_data['lumen_area'].max()
        self.report_data['elliptic_ratio'] = (
            self.report_data['elliptic_ratio'] / self.report_data['elliptic_ratio'].max()
        )
        self.report_data['elliptic_ratio'] = (self.report_data['elliptic_ratio'] - 1) * -1
        self.report_data['vector_angle'] = self.report_data['vector_angle'] / self.report_data['vector_angle'].max()
        self.report_data['vector_length'] = self.report_data['vector_length'] / self.report_data['vector_length'].max()
        self.report_data['vector'] = (self.report_data['vector_angle'] + self.report_data['vector_length']) / 2
        self.report_data['vector'] = (self.report_data['vector'] - 1) * -1

    def optimize_window_size_and_weights(self):
        combinations = self.lookup_table_weights_combinations()

        variance_maxima = []
        variance_minima = []
        weights_greater = []
        weights_lesser = []
        window_sizes = []
        systolic_idx = []
        diastolic_idx = []
        signals_systole = []
        signals_diastole = []

        progress = QProgressDialog(self.main_window)
        progress.setWindowFlags(Qt.Dialog)
        progress.setModal(True)
        progress.setMinimum(self.min_window_size)
        progress.setMaximum(self.max_window_size + 1)
        progress.resize(500, 100)
        progress.setValue(0)
        progress.setValue(1)
        progress.setValue(0)  # trick to make progress bar appear
        progress.setWindowTitle('Extracting systolic and diastolic frames...')
        progress.show()

        for window_size in range(self.min_window_size, self.max_window_size + 1):
            progress.setValue(window_size)
            if progress.wasCanceled():
                return False
            self.smooth(window_size=window_size)
            variance_greater = []
            local_maxima_indices = []
            signals_sys = []
            for weights in combinations:
                local_extrema_differences, local_extrema_indices, signal = self.variance_of_local_extrema_weights(
                    weights, np.greater
                )
                variance_greater.append(np.var(local_extrema_differences))
                local_maxima_indices.append(local_extrema_indices)
                signals_sys.append(signal)

            variance_lesser = []
            local_minima_indices = []
            signals_dia = []
            for weights in combinations:
                local_extrema_differences, local_extrema_indices, signal = self.variance_of_local_extrema_weights(
                    weights, np.less
                )
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
            window_sizes.append(window_size)
            systolic_idx.append(systolic_indices_loop)
            diastolic_idx.append(diastolic_indices_loop)
            signals_systole.append(signals_sys_min)
            signals_diastole.append(signals_dia_min)

        progress.close()

        # find the minimum variance in the result and return the index
        sum_var = [sum(x) for x in zip(variance_maxima, variance_minima)]

        # find min index and use this to find value at same index in window_size
        idx = np.argmin(sum_var)
        window = window_sizes[idx]
        weights_systole = weights_greater[idx]
        weigths_diastole = weights_lesser[idx]
        self.systolic_indices = systolic_idx[idx]
        self.diastolic_indices = diastolic_idx[idx]
        self.signal_systole = signals_systole[idx]
        self.signal_diastole = signals_diastole[idx]
        logger.info(f'Optimal window size: {window}')
        logger.info(
            f'\033[91mOptimal weights Systole:\033[0m\nLumen area: {weights_systole[0]}'
            f', Elliptic Ratio: {weights_systole[1]}, Vector: {weights_systole[2]}'
        )
        logger.info(
            f'\033[94mOptimal weights Diastole:\033[0m\nLumen area: {weigths_diastole[0]}'
            f', Elliptic Ratio: {weigths_diastole[1]}, Vector: {weigths_diastole[2]}'
        )

        return True

    def lookup_table_weights_combinations(self):
        combinations = []

        # Determine the range based on the self.step_size size
        num_values = int(1 / self.step_size)

        # Loop through all possible values for the first element
        for a in range(1, num_values + 1):
            a_value = a * self.step_size
            # Loop through all possible values for the second element
            for b in range(1, num_values + 1 - a):
                b_value = b * self.step_size
                # Calculate the third element
                c_value = 1 - a_value - b_value
                # Add the combination to the list if it sums to 1
                if c_value >= 0:
                    combinations.append([round(a_value, 2), round(b_value, 2), round(c_value, 2)])

        return combinations

    def smooth(self, window_size):
        self.lumen_area_smoothed = self.report_data['lumen_area'].rolling(window=window_size).mean()
        self.elliptic_ratio_smoothed = self.report_data['elliptic_ratio'].rolling(window=window_size).mean()
        self.vector_smoothed = self.report_data['vector'].rolling(window=window_size).mean()

    def variance_of_local_extrema_weights(self, weights, comparison_function):
        alpha, beta, gamma = weights
        signal = alpha * self.lumen_area_smoothed + beta * self.elliptic_ratio_smoothed + gamma * self.vector_smoothed

        local_extrema = argrelextrema(signal.values, comparison_function, order=5)
        local_extrema = local_extrema[0].tolist()
        local_extrema_indices = self.report_data.index[local_extrema].tolist()
        local_extrema_differences = [
            local_extrema_indices[i] - local_extrema_indices[i - 1] for i in range(1, len(local_extrema_indices))
        ]
        return local_extrema_differences, local_extrema_indices, signal

    def propagate_gating(self):
        sys_mean_diff = round(np.mean(np.diff(self.systolic_indices)))
        self.systolic_indices_plot = self.systolic_indices.copy()
        self.systolic_indices = (
            np.arange(0, min(self.systolic_indices), sys_mean_diff, dtype=int).tolist()
            + self.systolic_indices
            + np.arange(
                max(self.systolic_indices) + sys_mean_diff, self.main_window.images.shape[0], sys_mean_diff, dtype=int
            ).tolist()
        )
        dia_mean_diff = round(np.mean(np.diff(self.diastolic_indices)))
        self.diastolic_indices_plot = self.diastolic_indices.copy()
        self.diastolic_indices = (
            np.arange(0, min(self.diastolic_indices), dia_mean_diff, dtype=int).tolist()
            + self.diastolic_indices
            + np.arange(
                max(self.diastolic_indices) + dia_mean_diff, self.main_window.images.shape[0], dia_mean_diff, dtype=int
            ).tolist()
        )

    def update_main_window(self):
        self.main_window.data['phases'] = ['-'] * len(self.main_window.data['phases'])  # reset phases
        for frame in self.diastolic_indices:
            self.main_window.data['phases'][frame] = 'D'
        for frame in self.systolic_indices:
            self.main_window.data['phases'][frame] = 'S'
        self.main_window.gated_frames_sys = self.systolic_indices
        self.main_window.gated_frames_dia = self.diastolic_indices
        self.main_window.display_slider.set_gated_frames(self.diastolic_indices)

    def plot_results(self):
        # Plot frame on x-axis and elliptic ratio and lumen area on y-axis
        _, ax = plt.subplots()
        ax.plot(self.report_data['frame'], self.elliptic_ratio_smoothed, label='Elliptic Ratio')
        ax.plot(self.report_data['frame'], self.lumen_area_smoothed, label='Lumen area (mm²)')
        ax.plot(self.report_data['frame'], self.vector_smoothed, label='Vector')
        ax.plot(self.report_data['frame'], self.signal_systole, label='Signal Systole')
        ax.plot(self.report_data['frame'], self.signal_diastole, label='Signal Diastole')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Elliptic Ratio, Lumen area (mm²), and Signal')
        ax.set_title('Elliptic Ratio, Lumen area (mm²), and Signal by Frame')
        ax.legend()

        # find frames corresponding to row in self.systolic_indices and self.diastolic_indices
        frames_systole = self.report_data.loc[self.systolic_indices_plot, 'frame'].tolist()
        frames_diastole = self.report_data.loc[self.diastolic_indices_plot, 'frame'].tolist()
        signal_systole = [self.signal_systole[frame] for frame in self.systolic_indices_plot]
        signal_diastole = [self.signal_diastole[frame] for frame in self.diastolic_indices_plot]

        # Scatter plot for 'S' (local maxima) and 'D' (local minima)
        ax.scatter(frames_systole, signal_systole, color='red', marker='o', label='S')
        ax.scatter(frames_diastole, signal_diastole, color='blue', marker='o', label='D')

        plt.show()
