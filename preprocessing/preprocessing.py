import pydicom as dcm
import numpy as np

from loguru import logger


class PreProcessing:
    def __init__(self, images, frame_rate, speed) -> None:
        self.images = images
        self.frame_rate = frame_rate
        self.speed = speed

    def __call__(self):
        tags_dia = self.IVUS_gating_diastole()
        tags_sys, distance_frames = self.IVUS_gating_systole(tags_dia)
        dia, sys = self.stack_generator(tags_dia, tags_sys)

        return tags_dia, tags_sys, distance_frames

    def IVUS_gating_diastole(self):
        """Performs gating of IVUS images, based on the paper G. D. Maso Talou, 
          "Improving Cardiac Phase Extraction in IVUS Studies by Integration of Gating Methods," 
          in IEEE Transactions on Biomedical Engineering (Dec. 2015), doi: 10.1109/TBME.2015.2449232."""

        if len(self.images.shape) == 4:
            self.images = self.images[:, :, :, 0]

        num_images = self.images.shape[0]
        pullback_len = self.speed * (num_images - 1) / self.frame_rate  # first image is recorded instantly so no time delay

        # calculate features, one for correlation and one for gradient
        feat_corr = np.zeros((num_images - 1, 1))
        feat_grad = np.zeros((num_images - 1, 1))

        for i in range(num_images - 1):
            C = self.normxcorr(self.images[i, :, :], self.images[i + 1, :, :])
            feat_corr[i] = 1 - np.max(C)
            gradx, grady = np.gradient(self.images[i, :, :])
            gradmag = abs(np.sqrt(gradx**2 + grady**2))
            feat_grad[i] = -np.sum(gradmag)

        # normalize data
        feat_corr_plus = feat_corr - np.min(feat_corr)
        feat_grad_plus = feat_grad - np.min(feat_grad)

        feat_corr_norm = feat_corr_plus / np.sum(feat_corr_plus)
        feat_grad_norm = feat_grad_plus / np.sum(feat_grad_plus)

        alpha = 0.25
        signal = alpha * feat_corr_norm + (1 - alpha) * feat_grad_norm

        # determine Fs (sampling frequency)
        time = np.linspace(0, pullback_len / self.speed, num_images)
        Fs = num_images / np.max(time)
        NFFT = int(2 ** np.ceil(np.log2(np.abs(len(signal)))))
        sample_signal = np.fft.fft(signal, NFFT, 0) / len(signal)
        freq = Fs / 2 * np.linspace(0, 1, NFFT // 2 + 1)

        # in order to return correct amplitude fft must be divided by length of sample
        # remove non-physiological frequencies, chosen heart rate range is 40-170 bpm
        freq[freq < 0.67] = 0
        freq[freq > 2.83] = 0

        sample_signal_red = sample_signal[0 : NFFT // 2 + 1]
        sample_signal_red[freq == 0] = 0

        # determine maximum frequency component of ss
        max_freq_ss = np.argmax(np.abs(sample_signal_red))
        max_freq_ss = freq[max_freq_ss]

        # find cutoff frequency
        sigma = 0.4
        cutoff_freq = (1 + sigma) * max_freq_ss

        # construct low pass kernel (nyqu_freq is half of the transducer frame rate) Nyquist frequency max that can be represented
        nyqu_freq = Fs / 2
        tau = 25 / 46
        v = 21 / 46
        kernel_freq = ((cutoff_freq / nyqu_freq) * np.sinc((cutoff_freq * np.arange(1, num_images + 1)) / nyqu_freq)) * (
            tau - v * np.cos(2 * np.pi * (np.arange(1, num_images + 1) / num_images))
        )

        # determine low frequency signal
        s_low = np.convolve(signal[:, 0], kernel_freq)
        s_low = s_low[0 : len(signal)]  # take first half only

        # find first minimum in heartbeat
        hr_frames = int(np.round(Fs / max_freq_ss))  # heart rate in frames
        idx = np.argmin(s_low[0:hr_frames])
        tags_dia = []
        tags_dia.append(idx)
        k = 0

        while idx < (num_images - hr_frames):
            k = k + 1
            # based on known heart rate seach within 2 frames for local minimum
            # increase cutoff_freq to look at higher frequencies
            idx2 = idx + np.arange(hr_frames - 2, hr_frames + 3)
            # find local minimum
            idx2 = idx2[idx2 < num_images - 1]
            min_idx = np.argmin(s_low[idx2])
            idx = idx2[min_idx]
            tags_dia.append(idx)

        cutoff_freq = cutoff_freq + max_freq_ss
        j = 1
        s_low = np.expand_dims(s_low, 1)

        while cutoff_freq < nyqu_freq:
            # recalculate f with new cutoff_freq value
            kernel_freq = ((cutoff_freq / nyqu_freq) * np.sinc((cutoff_freq * np.arange(1, num_images + 1)) / nyqu_freq)) * (
                tau - v * np.cos(2 * np.pi * (np.arange(1, num_images + 1) / num_images))
            )
            # add extra columns to s_low to create surface as in paper
            s_temp = np.expand_dims(np.convolve(signal[:, 0], kernel_freq), 1)
            s_low = np.concatenate((s_low, s_temp[0 : len(signal)]), 1)

            # adjust each previous minimum p(i) to the nearest minimum
            for i in range(len(tags_dia)):
                # find index of new lowest p in +/-1 neighbour search
                search_index = np.arange(tags_dia[i] - 1, tags_dia[i] + 2)
                search_index = search_index[search_index >= 0]
                search_index = search_index[search_index < len(signal)]  # <=?
                search_index = np.in1d(np.arange(0, len(signal)), search_index)
                # determine index of min value in the specified neighbour range
                min_value = np.argmin(s_low[search_index, j])
                # switch from logical to indexed values
                search_index = np.argwhere(search_index)
                tags_dia[i] = search_index[min_value][0]
            # iteratively adjust to the new minimum
            # increase cutoff_freq to look at higher frequencies
            cutoff_freq = cutoff_freq + max_freq_ss
            j = j + 1

        # group images between p(i) and p(i+1)
        # output frames corresponding to each cardiac phase
        heartbeat = []
        for i in range(len(tags_dia) - 1):
            heartbeat.append(list(np.arange(tags_dia[i], tags_dia[i + 1])))

        return tags_dia

    def normxcorr(self, image1, image2):
        C = np.zeros_like(image1)
        image1_mean = np.mean(image1)
        image2_mean = np.mean(image2)
        image1_std = np.std(image1)
        image2_std = np.std(image2)
        C = np.sum((image1 - image1_mean) * (image2 - image2_mean)) / (image1_std * image2_std)
        C = C / (image1.shape[0] * image1.shape[1])
        return C

    def IVUS_gating_systole(self, tags_dia, num_frames_to_remove=0.25):
        """find the minimum signal value in between two diastolic frames and use this as the systolic frame"""
        s_low = self.IVUS_gating_diastole()
        
        num_frames = len(tags_dia)
        num_frames_to_ignore = int(np.round(num_frames * 0.25))
        num_frames_to_analyze = num_frames - 2 * num_frames_to_ignore
        
        tags_sys = []

        for i in range(num_frames - 1):
            if i < num_frames_to_ignore or i >= (num_frames - num_frames_to_ignore):
                continue  # Ignore the first and last 25% of frames
            
            start_idx = tags_dia[i] + num_frames_to_ignore
            end_idx = tags_dia[i + 1] - num_frames_to_ignore
            
            # Extract the relevant portion of s_low for analysis
            segment = s_low[start_idx:end_idx]
            
            # Find the index of the minimum value within the segment
            min_idx = start_idx + np.argmin(segment)
            tags_sys.append(min_idx)
        
        return tags_sys

    def stack_generator(self, tags_dia, tags_sys):
        """generate a stack for systolic and diastolic images based on list position"""
        diastole = self.images[tags_dia, :, :]
        systole = self.images[tags_sys, :, :]

        return diastole, systole
