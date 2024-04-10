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
        self.correlation = None
        self.blurring = None
        self.vertical_lines = []
        self.selected_line = None
        self.phases = []
        self.systolic_indices = []
        self.diastolic_indices = []

    def __call__(self):
        self.main_window.status_bar.showMessage('Contour-based gating...')
        self.report_data = report(self.main_window, suppress_messages=True)  # compute all needed data
        if self.report_data is None:
            ErrorMessage(self.main_window, 'Please ensure that a DICOM file was read and contours were drawn')
            self.main_window.status_bar.showMessage(self.main_window.waiting_status)
            return

        self.define_intramural_part()
        self.frames = self.crop_frames(x1=50, x2=450, y1=50, y2=450)
        self.shortest_distance = self.report_data['shortest_distance']
        self.vector_angle = self.report_data['vector_angle']
        self.vector_length = self.report_data['vector_length']
        self.prepare_data()
        success = self.plot_data()
        logger.debug(self.phases)
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
            self.report_data = self.report_data[self.report_data['frame'].between(lower_limit, upper_limit)]
            self.frames = self.main_window.images[lower_limit:upper_limit]

    def crop_frames(self, x1=50, x2=450, y1=50, y2=450):
        """Crops frames to a specific region."""
        cropped_frames = []
        for frame in self.frames:
            cropped_frame = frame[x1:x2, y1:y2]
            cropped_frames.append(cropped_frame)
        return cropped_frames

    def prepare_data(self):
        """Prepares data for plotting."""
        self.correlation = self.normalize_data(self.calculate_correlation())
        self.blurring = self.normalize_data(self.calculate_blurring_fft())
        self.shortest_distance = self.normalize_data(self.shortest_distance)
        self.vector_angle = self.normalize_data(self.vector_angle)
        self.vector_length = self.normalize_data(self.vector_length)
        print("Data prepared successfully")
        
    def normalize_data(self, data):
        return (data - np.min(data)) / np.sum(data - np.min(data))

    def calculate_correlation(self):
        """Calculates correlation coefficients between consecutive frames."""
        correlations = []
        for i in range(len(self.frames) - 1):
            corr = np.corrcoef(self.frames[i].ravel(), self.frames[i + 1].ravel())[0, 1]
            correlations.append(corr)
        return correlations

    def calculate_blurring_fft(self):
        """Calculates blurring using Fast Fourier Transform. Take average of the 90% highest frequencies."""
        blurring_scores = []
        for frame in self.frames:
            fft_data = np.fft.fft2(frame)
            fft_shifted = np.fft.fftshift(fft_data)
            magnitude_spectrum = np.abs(fft_shifted)
            sorted_magnitude_spectrum = np.sort(magnitude_spectrum.ravel())
            threshold = int(0.9 * len(sorted_magnitude_spectrum))
            blurring_score = np.mean(sorted_magnitude_spectrum[threshold:])
            blurring_scores.append(blurring_score)
        return blurring_scores

    def identify_extrema(self, signal):
        maxima_indices = argrelextrema(signal, np.greater)[0]
        minima_indices = argrelextrema(signal, np.less)[0]

        # Combine maxima and minima indices into one array
        extrema_indices = np.concatenate((maxima_indices, minima_indices))
        extrema_indices = np.sort(extrema_indices)

        return extrema_indices, maxima_indices

    def combined_signal(self, signal_list, window_size=5, maxima_only=False):
        # Find the minimum length among all signals
        min_length = min(signal.shape[0] for signal in signal_list)

        # Trim signals to the minimum length
        for i, signal in enumerate(signal_list):
            signal_list[i] = signal[:min_length]

        # smooth_curve for all signals
        smoothed_signals = []
        for signal in signal_list:
            smoothed_signal = self.smooth_curve(signal, window_size=window_size)
            smoothed_signals.append(smoothed_signal)

        # find extrema indices for all curves
        extrema_indices = []
        for signal in smoothed_signals:
            if maxima_only:
                extrema_indices.append(self.identify_extrema(signal)[1])
            else:
                extrema_indices.append(self.identify_extrema(signal)[0])

        # find variability in extrema indices
        variability = []
        for extrema in extrema_indices:
            variability.append(np.std(np.diff(extrema)))

        # calculate sum of all variabilities and then create a combined signal with weights as percent of variability
        sum_variability = np.sum(variability)
        weights = [(var / sum_variability) ** -1 for var in variability]

        combined_signal = np.zeros(len(signal_list[0]))
        for i, signal in enumerate(signal_list):
            combined_signal += weights[i] * signal

        return combined_signal
    
    def smooth_curve(self, signal, window_size=5):
        return np.convolve(signal, np.ones(window_size) / window_size, mode='valid')

    def on_click(self, event):
        if event.inaxes:
            # Check if there are existing lines
            if not self.vertical_lines:
                new_line = plt.axvline(x=event.xdata, color='r', linestyle='--')
                self.vertical_lines.append(new_line)
                plt.draw()
            else:
                # Check if click is near any existing line
                distances = [abs(line.get_xdata()[0] - event.xdata) for line in self.vertical_lines]
                min_distance = min(distances)
                if min_distance < 5:  # You can adjust the sensitivity for line selection
                    self.selected_line = self.vertical_lines[np.argmin(distances)]
                else:
                    new_line = plt.axvline(x=event.xdata, color='r', linestyle='--')
                    self.vertical_lines.append(new_line)
                plt.draw()

        logger.debug(self.vertical_lines)

    def on_release(self, event):
        self.selected_line = None

    def on_motion(self, event):
        if self.selected_line:
            self.selected_line.set_xdata([event.xdata] * 2)
            plt.draw()

    def on_close(self, event):
        plt.close()

    def get_x_indices(self):
        x_indices = [line.get_xdata()[0] for line in self.vertical_lines]
        return x_indices

    def plot_data(self):
        signal_list_max = [
            self.smooth_curve(self.correlation),
            self.smooth_curve(self.blurring),
        ]

        singal_list_extrema = [
            self.smooth_curve(self.shortest_distance),
            self.smooth_curve(self.vector_angle),
            self.smooth_curve(self.vector_length),
        ]

        # s_max_w2 = self.combined_signal(signal_list_max, window_size=2, maxima_only=True)
        s_max_w5 = self.combined_signal(signal_list_max, window_size=5, maxima_only=True)
        # s_max_w10 = self.combined_signal(signal_list_max, window_size=10, maxima_only=True)
        # s_extrema_w2 = self.combined_signal(singal_list_extrema, window_size=15, maxima_only=False)
        s_extrema_w5 = self.combined_signal(singal_list_extrema, window_size=5, maxima_only=False)
        # s_extrema_w10 = self.combined_signal(singal_list_extrema, window_size=10, maxima_only=False)

        # check mean difference between s_max and s_extrema curves, and scale smaller curve to match the larger one
        # for combined signal
        # mean_max_values = np.mean([s_max_w5, s_max_w10, s_max_w2])
        # mean_extrema_values = np.mean([s_extrema_w5, s_extrema_w10, s_extrema_w2])

        mean_max_values = np.mean(s_max_w5)
        mean_extrema_values = np.mean(s_extrema_w5)

        factor_diff = mean_max_values / mean_extrema_values

        if factor_diff < 1:
            # s_extrema_w2 = s_extrema_w2 * factor_diff
            s_extrema_w5 = s_extrema_w5 * factor_diff
            # s_extrema_w10 = s_extrema_w10 * factor_diff
        else:
            # s_max_w2 = s_max_w2 * factor_diff
            s_max_w5 = s_max_w5 * factor_diff
            # s_max_w10 = s_max_w10 * factor_diff

        plt.figure()

        # Plot your data
        # plt.plot(s_max_w2, color='r')
        plt.plot(s_max_w5, color='r')
        # plt.plot(s_max_w10, color='r')
        # plt.plot(s_extrema_w2, color='b')
        plt.plot(s_extrema_w5, color='b')
        # plt.plot(s_extrema_w10, color='b')
        # plt.legend(['s_max_w2', 's_max_w5', 's_max_w10', 's_extrema_w2', 's_extrema_w5', 's_extrema_w10'])
        plt.plot(singal_list_extrema[0], color='grey')
        plt.plot(singal_list_extrema[1], color='grey')
        plt.plot(singal_list_extrema[2], color='grey')
        plt.legend(['s_max_w5', 's_extrema_w5', 'shortest_distance', 'vector_angle', 'vector_length'])

        # Connect the event handlers
        plt.connect('button_press_event', self.on_click)
        plt.connect('motion_notify_event', self.on_motion)
        plt.connect('button_release_event', self.on_release)
        plt.connect('close_event', self.on_close)

        plt.show(block=True)

        self.phases = self.get_x_indices()
        self.phases = [round(phase, 0) for phase in self.phases]

        return True

    def identify_systole_diastole(self):
        # split self.phases by every second element
        first_indices = self.phases[::2]
        second_indices = self.phases[1::2]

        first_elliptic_ratio = np.mean(self.report_data['frame'][first_indices])

        pass

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
