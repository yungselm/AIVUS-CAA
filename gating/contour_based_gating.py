import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from loguru import logger
from scipy.signal import argrelextrema, find_peaks, butter, filtfilt

from gui.popup_windows.message_boxes import ErrorMessage
from gui.popup_windows.frame_range_dialog import FrameRangeDialog
from gui.right_half.right_half import toggle_diastolic_frame, toggle_systolic_frame
from report.report import report


class ContourBasedGating:
    def __init__(self, main_window, lowcut=0.6667, highcut=3.0, fs=30.0, order=4):
        self.main_window = main_window
        self.intramural_threshold = main_window.config.gating.intramural_threshold
        # Filter parameters
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.correlation = None
        self.blurring = None
        self.vertical_lines = []
        self.selected_line = None
        self.current_phase = None
        self.tmp_phase = None
        self.frame_marker = None
        self.phases = []
        self.systolic_indices = []
        self.diastolic_indices = []
        self.default_line_color = 'grey'
        self.default_linestyle = (0, (1, 3))

    def __call__(self):
        self.main_window.status_bar.showMessage('Contour-based gating...')
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
            self.report_data = report(
                self.main_window, lower_limit, upper_limit, suppress_messages=True
            )  # compute all needed data
            if self.report_data is None:
                ErrorMessage(self.main_window, 'Please ensure that an input file was read and contours were drawn')
                self.main_window.status_bar.showMessage(self.main_window.waiting_status)
                return False

            if (
                lower_limit == 0 and upper_limit == self.main_window.images.shape[0]
            ):  # automatic detection of intramural part
                mean_elliptic_ratio = self.report_data['elliptic_ratio'].rolling(window=5, closed='both').mean()

            if len(self.report_data) != upper_limit - lower_limit:
                missing_frames = [
                    frame
                    for frame in range(lower_limit + 1, upper_limit + 1)
                    if frame not in self.report_data['frame'].values
                ]
                str_missing = self.connect_consecutive_frames(missing_frames)
                ErrorMessage(self.main_window, f'Please add contours to frames {str_missing}')
                return False
            self.frames = self.main_window.images[lower_limit:upper_limit]
            self.x = self.report_data['frame'].values  # want 1-based indexing for GUI
            return True
        return False

    def connect_consecutive_frames(self, missing: list) -> str:
        nums = sorted(set(missing))
        connected = []
        i = 0
        while i < len(nums):
            j = i
            while j < len(nums) - 1 and nums[j + 1] - nums[j] == 1:
                j += 1
            if i == j:
                connected.append([nums[i]])
            else:
                connected.append(nums[i : j + 1])
            i = j + 1
        connected = [
            (f'{sublist[0]}-{sublist[-1]}' if len(sublist) > 2 else ", ".join(map(str, sublist)))
            for sublist in connected
        ]
        return ", ".join(connected)

    def crop_frames(self, x1=50, x2=450, y1=50, y2=450):
        """Crops frames to a specific region."""
        self.frames = self.frames[:, x1:x2, y1:y2]

    def prepare_data(self):
        """Prepares data for plotting."""
        # Normalize signals
        self.correlation_nor = self.normalize_data(self.calculate_correlation())
        self.blurring_nor = self.normalize_data(self.calculate_blurring_fft())
        self.shortest_distance_nor = self.normalize_data(self.shortest_distance)
        self.vector_angle_nor = self.normalize_data(self.vector_angle)
        self.vector_length_nor = self.normalize_data(self.vector_length)

        # Apply bandpass filter to each normalized signal
        self.correlation = self.bandpass_filter(self.correlation_nor)
        self.blurring = self.bandpass_filter(self.blurring_nor)
        self.shortest_distance = self.bandpass_filter(self.shortest_distance_nor)
        self.vector_angle = self.bandpass_filter(self.vector_angle_nor)
        self.vector_length = self.bandpass_filter(self.vector_length_nor)

        print("Data prepared and bandpass filtered successfully")

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
        """Calculates blurring using Fast Fourier Transform. Take average of the 10% highest frequencies."""
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

    def bandpass_filter(self, signal):
        """
        Applies a Butterworth bandpass filter to the input signal using instance parameters.

        Parameters:
        - signal (array-like): The input signal to filter.

        Returns:
        - filtered_signal (numpy.ndarray): The bandpass filtered signal.
        """
        nyquist = 0.5 * self.fs
        low = self.lowcut / nyquist
        high = self.highcut / nyquist

        # Design Butterworth bandpass filter
        b, a = butter(self.order, [low, high], btype='band')

        # Apply filter using filtfilt for zero phase distortion
        filtered_signal = filtfilt(b, a, signal)

        return filtered_signal
    
    # def test_plot(self, raw_signals, smoothed_signals, extrema_indices):
    #     # function that plots signals not smoothed, signals smoothed and extrema
    #     fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    #     for i, signal in enumerate(raw_signals):
    #         ax[0].plot(signal, label=f'Raw signal {i + 1}')
    #     for i, signal in enumerate(smoothed_signals):
    #         ax[1].plot(signal, label=f'Smoothed signal {i + 1}')
    #     for i, extrema in enumerate(extrema_indices):
    #         ax[1].plot(extrema, smoothed_signals[i][extrema], 'x', label=f'Extrema {i + 1}')
    #     ax[0].set_title('Raw signals')
    #     ax[1].set_title('Smoothed signals')
    #     ax[0].legend()
    #     ax[1].legend()
    #     # save plot in testing folder
    #     plt.savefig('testing/combined_signal.png')

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

        # self.test_plot(signal_list, smoothed_signals, extrema_indices)

        # find variability in extrema indices, based on assumption that heartrate is regular
        variability = []
        for extrema in extrema_indices:
            variability.append(np.std(np.diff(extrema)))

        # calculate sum of all variabilities and then create a combined signal with weights as percent of variability
        sum_variability = np.sum(variability)
        weights = [(var / sum_variability) ** -1 for var in variability]

        combined_signal = np.zeros(len(signal_list[0]))
        for i, signal in enumerate(signal_list):
            combined_signal += weights[i] * signal
        
        # if name of signal list is signal_list_max then create a combined signal with the _nor otherwise signal_list_extrema
        combined_signal_nor = np.zeros(len(signal_list[0]))

        if maxima_only == True:
            signal_list_max = [
                self.correlation_nor,
                self.blurring_nor,
            ]
            for i, signal in enumerate(signal_list_max):
                combined_signal_nor += weights[i] * signal
        else:  # for extrema
            signal_list_extrema = [
                self.shortest_distance_nor,
                self.vector_angle_nor,
                self.vector_length_nor,
            ]
            for i, signal in enumerate(signal_list_extrema):
                combined_signal_nor += weights[i] * signal

        return combined_signal, combined_signal_nor

    def identify_extrema(self, signal):
        # Remove NaN and infinite values from the signal
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

        # Dynamically calculate prominence based on the signal's characteristics
        min_height = np.percentile(signal, 50)  # Only pick peaks above the 90th percentile

        # Find maxima and minima using find_peaks with dynamic prominence
        maxima_indices, _ = find_peaks(signal, distance=12, height=min_height)
        minima_indices, _ = find_peaks(-signal, distance=12, height=min_height)

        # Combine maxima and minima indices into one array and sort them
        extrema_indices = np.concatenate((maxima_indices, minima_indices))
        extrema_indices = np.sort(extrema_indices)

        return extrema_indices, maxima_indices
    
    def smooth_curve(self, signal, window_size=5):
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

    def plot_data(self):
        signal_list_max = [
            self.correlation,
            self.blurring,
        ]

        signal_list_extrema = [
            self.shortest_distance,
            self.vector_angle,
            self.vector_length,
        ]
        s_max_w5, s_max_w5_nor = self.combined_signal(signal_list_max, window_size=5, maxima_only=True)
        s_extrema_w5, s_extrema_w5_nor = self.combined_signal(signal_list_extrema, window_size=5, maxima_only=False)

        mean_max_values = np.mean(s_max_w5)
        mean_extrema_values = np.mean(s_extrema_w5)

        factor_diff = mean_max_values / mean_extrema_values

        if factor_diff < 1:
            s_extrema_w5 = s_extrema_w5 * factor_diff
            s_extrema_w5_nor = s_extrema_w5_nor * factor_diff
        else:
            s_max_w5 = s_max_w5 * factor_diff
            s_max_w5_nor = s_max_w5_nor * factor_diff

        # Adjust the scaling in steps of 100 frames
        step_size = 100
        for i in range(0, len(s_extrema_w5), step_size):
            end_idx = min(i + step_size, len(s_extrema_w5))
            range_extrema = np.max(s_extrema_w5[i:end_idx]) - np.min(s_extrema_w5[i:end_idx])
            range_max = np.max(s_max_w5[i:end_idx]) - np.min(s_max_w5[i:end_idx])
            # Scale smaller range to larger range within the step
            if range_extrema < range_max:
                s_extrema_w5[i:end_idx] = s_extrema_w5[i:end_idx] * (range_max / range_extrema)
            else:
                s_max_w5[i:end_idx] = s_max_w5[i:end_idx] * (range_extrema / range_max)
        
        for i in range(0, len(s_extrema_w5_nor), step_size):
            end_idx = min(i + step_size, len(s_extrema_w5_nor))
            range_extrema = np.max(s_extrema_w5_nor[i:end_idx]) - np.min(s_extrema_w5_nor[i:end_idx])
            range_max = np.max(s_max_w5_nor[i:end_idx]) - np.min(s_max_w5_nor[i:end_idx])
            # Scale smaller range to larger range within the step
            if range_extrema < range_max:
                s_extrema_w5_nor[i:end_idx] = s_extrema_w5_nor[i:end_idx] * (range_max / range_extrema)
            else:
                s_max_w5_nor[i:end_idx] = s_max_w5_nor[i:end_idx] * (range_extrema / range_max)

        self.fig = self.main_window.gating_display.fig
        self.fig.clear()
        self.ax = self.fig.add_subplot()

        self.ax.plot(self.x, s_max_w5, color='green', label='Image based gating')
        self.ax.plot(self.x, s_extrema_w5, color='yellow', label='Contour based gating')
        self.ax.plot(self.x, self.smooth_curve(s_max_w5_nor) - .01, color='green', linestyle='dashed', label='Image based gating (unfiltered)')
        self.ax.plot(self.x, self.smooth_curve(s_extrema_w5_nor) - .01, color='yellow', linestyle='dashed', label='Contour based gating (unfiltered)')
        # self.ax.plot(self.x, signal_list_extrema[0] - 0.001, color='grey', label='_hidden')
        # self.ax.plot(self.x, signal_list_extrema[1] - 0.001, color='grey', label='_hidden')
        # self.ax.plot(self.x, signal_list_extrema[2] - 0.001, color='grey', label='_hidden')
        self.ax.set_xlabel('Frame')
        self.ax.get_yaxis().set_visible(False)
        legend = self.ax.legend(ncol=2, loc='upper right')
        legend.set_draggable(True)

        plt.connect('button_press_event', self.on_click)
        plt.connect('motion_notify_event', self.on_motion)
        plt.connect('button_release_event', self.on_release)

        # Draw lines
        if not self.main_window.gated_frames_dia and not self.main_window.gated_frames_sys:
            self.automatic_gating(s_max_w5, s_extrema_w5)
            self.draw_existing_lines(self.main_window.gated_frames_dia, self.main_window.diastole_color_plt)
            self.draw_existing_lines(self.main_window.gated_frames_sys, self.main_window.systole_color_plt)
        else:
            self.draw_existing_lines(self.main_window.gated_frames_dia, self.main_window.diastole_color_plt)
            self.draw_existing_lines(self.main_window.gated_frames_sys, self.main_window.systole_color_plt)

        plt.tight_layout()
        plt.draw()

        return True

    # def plot_data(self):
    #     signal_list_max = [self.correlation, self.blurring]
    #     signal_list_extrema = [self.shortest_distance, self.vector_angle, self.vector_length]

    #     # Calculate combined signals with window size 5
    #     s_max_w5, s_max_w5_nor = self.combined_signal(signal_list_max, window_size=5, maxima_only=True)
    #     s_extrema_w5, s_extrema_w5_nor = self.combined_signal(signal_list_extrema, window_size=5, maxima_only=False)

    #     # Adjust scaling between max and extrema signals
    #     mean_max_values = np.mean(s_max_w5)
    #     mean_extrema_values = np.mean(s_extrema_w5)
    #     factor_diff = mean_max_values / mean_extrema_values
    #     mean_max_values_nor = np.mean(s_max_w5_nor)
    #     mean_extrema_values_nor = np.mean(s_extrema_w5_nor)
    #     factor_diff_nor = mean_max_values_nor / mean_extrema_values_nor

    #     if factor_diff < 1:
    #         s_extrema_w5 *= factor_diff
    #         s_extrema_w5_nor *= factor_diff_nor
    #     else:
    #         s_max_w5 *= factor_diff
    #         s_max_w5_nor *= factor_diff_nor

    #     # Step scaling adjustment for smoothing
    #     step_size = 100
    #     for i in range(0, len(s_extrema_w5), step_size):
    #         end_idx = min(i + step_size, len(s_extrema_w5))
    #         range_extrema = np.max(s_extrema_w5[i:end_idx]) - np.min(s_extrema_w5[i:end_idx])
    #         range_max = np.max(s_max_w5[i:end_idx]) - np.min(s_max_w5[i:end_idx])
    #         scale_factor = range_max / range_extrema if range_extrema < range_max else range_extrema / range_max
    #         s_extrema_w5[i:end_idx] *= scale_factor
    #         s_max_w5[i:end_idx] *= scale_factor

    #     # Plot the signals
    #     self.fig = self.main_window.gating_display.fig
    #     self.fig.clear()
    #     self.ax = self.fig.add_subplot()

    #     self.ax.plot(self.x, s_max_w5, color='green', label='Image based gating')
    #     self.ax.plot(self.x, s_extrema_w5, color='yellow', label='Contour based gating')
    #     self.ax.plot(self.x, s_max_w5_nor, color='green', linestyle='dashed', label='Image based gating (unfiltered)')
    #     self.ax.plot(self.x, s_extrema_w5_nor, color='yellow', linestyle='dashed', label='Contour based gating (unfiltered)')

    #     # Additional setup for plot
    #     self.ax.set_xlabel('Frame')
    #     self.ax.get_yaxis().set_visible(False)
    #     legend = self.ax.legend(ncol=2, loc='upper right')
    #     legend.set_draggable(True)

    #     plt.connect('button_press_event', self.on_click)
    #     plt.connect('motion_notify_event', self.on_motion)
    #     plt.connect('button_release_event', self.on_release)
        
    #     # Draw lines
    #     if not self.main_window.gated_frames_dia and not self.main_window.gated_frames_sys:
    #         self.automatic_gating(s_max_w5, s_extrema_w5)
    #         self.draw_existing_lines(self.main_window.gated_frames_dia, self.main_window.diastole_color_plt)
    #         self.draw_existing_lines(self.main_window.gated_frames_sys, self.main_window.systole_color_plt)
    #     else:
    #         self.draw_existing_lines(self.main_window.gated_frames_dia, self.main_window.diastole_color_plt)
    #         self.draw_existing_lines(self.main_window.gated_frames_sys, self.main_window.systole_color_plt)

    #     plt.tight_layout()
    #     plt.draw()

    #     return True
    
    def automatic_gating(self, maxima_signal, extrema_signal, threshold=5):
        # Identify local maxima and minima
        maxima_indices = self.identify_extrema(maxima_signal)[0]
        extrema_indices = list(self.identify_extrema(extrema_signal)[0])  # Convert to list to allow removals

        gated_indices = []

        # Check for all indices in maxima and extrema if they are not more than 5 frames apart
        for maxima in maxima_indices:
            close_extrema = [extrema for extrema in extrema_indices if abs(maxima - extrema) <= threshold]
            
            if close_extrema:
                # Select the first valid extrema that is within the threshold distance
                closest_extrema = close_extrema[0]
                gated_indices.append(round((maxima + closest_extrema) / 2))
                extrema_indices.remove(closest_extrema)  # Remove to avoid duplicate gating

        # Split by every second frame, diastole frames are where lumen_area is greater in sum than the other half
        first_half = gated_indices[::2]
        second_half = gated_indices[1::2]

        sum_first_half = sum([self.report_data.loc[self.report_data['frame'] == frame, 'lumen_area'].values[0] for frame in first_half])
        sum_second_half = sum([self.report_data.loc[self.report_data['frame'] == frame, 'lumen_area'].values[0] for frame in second_half])

        if sum_first_half > sum_second_half:
            self.main_window.gated_frames_dia = first_half
            self.main_window.gated_frames_sys = second_half
        else:
            self.main_window.gated_frames_dia = second_half
            self.main_window.gated_frames_sys = first_half
            
        for frame in self.main_window.gated_frames_dia:
            self.main_window.data['phases'][frame] = 'D'
        for frame in self.main_window.gated_frames_sys:
            self.main_window.data['phases'][frame] = 'S'
        
    def on_click(self, event):
        if self.fig.canvas.cursor().shape() != 0:  # zooming or panning mode
            return
        if event.button is MouseButton.LEFT and event.inaxes:
            new_line = True
            set_dia = False
            set_sys = False
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
                if self.current_phase == 'D':
                    color = self.main_window.diastole_color_plt
                    set_dia = True
                elif self.current_phase == 'S':
                    color = self.main_window.systole_color_plt
                    set_sys = True
                else:
                    color = self.default_line_color
                self.selected_line = plt.axvline(
                    x=event.xdata, color=color, linestyle=self.default_linestyle
                )
                self.vertical_lines.append(self.selected_line)

            self.selected_line.set_linestyle('dashed')
            plt.draw()

            set_slider_to = round(set_slider_to - 1)  # slider is 0-based
            self.main_window.display_slider.set_value(set_slider_to, reset_highlights=False)

            if set_slider_to in self.main_window.gated_frames_dia or set_dia:
                self.tmp_phase = 'D'
                toggle_diastolic_frame(self.main_window, False, drag=True)
            elif set_slider_to in self.main_window.gated_frames_sys or set_sys:
                self.tmp_phase = 'S'
                toggle_systolic_frame(self.main_window, False, drag=True)

    def on_release(self, event):
        if self.fig.canvas.cursor().shape() != 0:  # zooming or panning mode
            return
        if event.button is MouseButton.LEFT and event.inaxes:
            if self.tmp_phase == 'D':
                self.main_window.diastolic_frame_box.setChecked(True)
                toggle_diastolic_frame(self.main_window, True, drag=True)
            elif self.tmp_phase == 'S':
                self.main_window.systolic_frame_box.setChecked(True)
                toggle_systolic_frame(self.main_window, True, drag=True)

        self.tmp_phase = None

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
            else:
                self.vertical_lines.remove(self.selected_line)
                self.selected_line = None
                self.tmp_phase = None
                plt.draw()

    def set_frame(self, frame):
        plt.autoscale(False)
        if self.frame_marker:
            self.frame_marker[0].remove()
        self.frame_marker = self.ax.plot(frame + 1, self.ax.get_ylim()[0], 'yo', clip_on=False)
        plt.draw()

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
