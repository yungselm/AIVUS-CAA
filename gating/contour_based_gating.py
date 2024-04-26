import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from loguru import logger
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtCore import Qt
from scipy.signal import argrelextrema

from gui.popup_windows.message_boxes import ErrorMessage
from gui.popup_windows.frame_range_dialog import FrameRangeDialog
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
        self.default_line_color = 'grey'
        self.default_linestyle = (0, (1, 3))

    def __call__(self):
        self.main_window.status_bar.showMessage('Contour-based gating...')
        self.report_data = report(self.main_window, suppress_messages=True)  # compute all needed data
        if self.report_data is None:
            ErrorMessage(self.main_window, 'Please ensure that an input file was read and contours were drawn')
            self.main_window.status_bar.showMessage(self.main_window.waiting_status)
            return

        dialog_success = self.define_intramural_part()
        if not dialog_success:
            self.main_window.status_bar.showMessage(self.main_window.waiting_status)
            return
        self.shortest_distance = self.report_data['shortest_distance']
        self.vector_angle = self.report_data['vector_angle']
        self.vector_length = self.report_data['vector_length']
        self.crop_frames(x1=50, x2=450, y1=50, y2=450)
        self.prepare_data()
        self.plot_data()
        # self.plot_results()

        self.main_window.status_bar.showMessage(self.main_window.waiting_status)

    def define_intramural_part(self):
        dialog = FrameRangeDialog(self.main_window)
        if dialog.exec_():
            lower_limit, upper_limit = dialog.getInputs()
            if (
                lower_limit == 0 and upper_limit == self.main_window.images.shape[0]
            ):  # automatic detection of intramural part
                mean_elliptic_ratio = self.report_data['elliptic_ratio'].rolling(window=5, closed='both').mean()
            self.report_data = self.report_data[
                self.report_data['frame'].between(lower_limit + 1, upper_limit, inclusive='both')
            ]
            if len(self.report_data) != upper_limit - lower_limit:
                missing_frames = [
                    str(frame)
                    for frame in range(lower_limit + 1, upper_limit + 1)
                    if frame not in self.report_data['frame'].values
                ]
                ErrorMessage(self.main_window, f'Please add contours to frames {", ".join(missing_frames)}')
                return False
            self.frames = self.main_window.images[lower_limit:upper_limit]
            self.x = self.report_data['frame'].values  # want 1-based indexing for GUI
            return True
        return False

    def crop_frames(self, x1=50, x2=450, y1=50, y2=450):
        """Crops frames to a specific region."""
        self.frames = self.frames[:, x1:x2, y1:y2]

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
        correlations.append(0)  # to match the length of the frames
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
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

    def plot_data(self):
        signal_list_max = [
            self.smooth_curve(self.correlation),
            self.smooth_curve(self.blurring),
        ]

        signal_list_extrema = [
            self.smooth_curve(self.shortest_distance),
            self.smooth_curve(self.vector_angle),
            self.smooth_curve(self.vector_length),
        ]

        s_max_w5 = self.combined_signal(signal_list_max, window_size=5, maxima_only=True)
        s_extrema_w5 = self.combined_signal(signal_list_extrema, window_size=5, maxima_only=False)

        mean_max_values = np.mean(s_max_w5)
        mean_extrema_values = np.mean(s_extrema_w5)

        factor_diff = mean_max_values / mean_extrema_values

        if factor_diff < 1:
            s_extrema_w5 = s_extrema_w5 * factor_diff
        else:
            s_max_w5 = s_max_w5 * factor_diff

        self.fig = self.main_window.gating_display.fig
        self.fig.clear()
        self.ax = self.fig.add_subplot()

        self.ax.plot(self.x, s_max_w5, color='green', label='Maxima')
        self.ax.plot(self.x, s_extrema_w5, color='yellow', label='Extrema')
        self.ax.plot(self.x, signal_list_extrema[0], color='grey', label='_hidden')
        self.ax.plot(self.x, signal_list_extrema[1], color='grey', label='_hidden')
        self.ax.plot(self.x, signal_list_extrema[2], color='grey', label='_hidden')
        self.ax.set_xlabel('Frame')
        self.ax.get_yaxis().set_visible(False)
        legend = self.ax.legend(ncol=2, loc='upper right')
        legend.set_draggable(True)

        plt.connect('button_press_event', self.on_click)
        plt.connect('motion_notify_event', self.on_motion)
        plt.connect('button_release_event', self.on_release)

        self.draw_existing_lines(self.main_window.gated_frames_dia, self.main_window.diastole_color_plt)
        self.draw_existing_lines(self.main_window.gated_frames_sys, self.main_window.systole_color_plt)

        plt.tight_layout()
        plt.draw()

        return True

    def on_click(self, event):
        if self.fig.canvas.cursor().shape() != 0:  # zooming or panning mode
            return
        if event.button is MouseButton.LEFT and event.inaxes:
            new_line = True
            set_slider_to = event.xdata
            if self.selected_line is not None:
                self.selected_line.set_linestyle(self.default_linestyle)
                self.selected_line = None
            if self.vertical_lines:
                # Check if click is near any existing line
                distances = [abs(line.get_xdata()[0] - event.xdata) for line in self.vertical_lines]
                if min(distances) < len(self.frames) / 100:  # sensitivity for line selection
                    self.selected_line = self.vertical_lines[np.argmin(distances)]
                    new_line = False
                    set_slider_to = self.selected_line.get_xdata()[0]
            if new_line:
                self.selected_line = plt.axvline(
                    x=event.xdata, color=self.default_line_color, linestyle=self.default_linestyle
                )
                self.vertical_lines.append(self.selected_line)

            self.selected_line.set_linestyle('dashed')
            plt.draw()

            self.main_window.display_slider.set_value(
                round(set_slider_to - 1), reset_highlights=False
            )  # slider is 0-based

    def on_release(self, event):
        pass

    def on_motion(self, event):
        if self.fig.canvas.cursor().shape() != 0:  # zooming or panning mode
            return
        if event.button is MouseButton.LEFT and self.selected_line:
            self.selected_line.set_xdata(event.xdata)
            if event.xdata is not None:
                self.main_window.display_slider.set_value(
                    round(event.xdata - 1), reset_highlights=False
                )  # slider is 0-based
                plt.draw()

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

    def draw_existing_lines(self, frames, color):
        frames = [frame for frame in frames if frame in (self.x - 1)]  # remove frames outside of user-defined range
        for frame in frames:
            self.vertical_lines.append(plt.axvline(x=frame + 1, color=color, linestyle=self.default_linestyle))

    def remove_lines(self):
        for line in self.vertical_lines:
            line.remove()
        self.vertical_lines = []
        plt.draw

    def update_color(self, color=None):
        color = color or self.default_line_color
        if self.selected_line is not None:
            self.selected_line.set_color(color)
            plt.draw()

    def reset_highlights(self):
        if self.selected_line is not None:
            self.selected_line.set_linestyle(self.default_linestyle)
            self.selected_line = None
            plt.draw()
