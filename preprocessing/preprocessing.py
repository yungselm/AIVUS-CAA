import pydicom as dcm 
import numpy as np 
 
from loguru import logger 
 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
 
class PreProcessing: 
    def __init__(self, images, frame_rate, speed) -> None: 
        self.images = images 
        self.frame_rate = frame_rate 
        self.speed = speed 
        self.num_images = None 
        self.Fs = None 
        self.signal = None 
        self.sample_signal = None 
        self.NFFT = None 
        print('__init__ done') 
 
    def __call__(self): 
        tags_dia = self.IVUS_gating_diastole() 
        tags_sys, distance_frames = self.IVUS_gating_systole() 
        # dia, sys = self.stack_generator(tags_dia, tags_sys) 
        print('Calling done') 
        return tags_dia, tags_sys, distance_frames 
 
    def IVUS_gating_diastole(self): 
        """Performs gating of IVUS images, based on the paper G. D. Maso Talou,  
          "Improving Cardiac Phase Extraction in IVUS Studies by Integration of Gating Methods,"  
          in IEEE Transactions on Biomedical Engineering (Dec. 2015), doi: 10.1109/TBME.2015.2449232.""" 
 
        if len(self.images.shape) == 4: 
            self.images = self.images[:, :, :, 0] 
 
        self.num_images = self.images.shape[0] 
        pullback_len = self.speed * (self.num_images - 1) / self.frame_rate  # first image is recorded instantly so no time delay 
 
        # calculate features, one for correlation and one for gradient 
        feat_corr = np.zeros((self.num_images - 1, 1)) 
        feat_grad = np.zeros((self.num_images - 1, 1)) 
 
        for i in range(self.num_images - 1): 
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
 
        #find the optimal weighting factor alpha for the two features by minimizing the standard deviation of the signal 
        alpha_new = np.arange(0.01, 1, 0.01)  
        sd_new = np.zeros(len(alpha_new)) 
         
        for i in np.arange(0,len(alpha_new),1):             
            signal = alpha_new[i] * feat_corr_norm + (1 - alpha_new[i]) * feat_grad_norm 
            sd_new[i] = np.std(signal) 
 
        alpha_min_idx = np.argmin(sd_new) 
        alpha_min = alpha_new[alpha_min_idx] 
 
        self.signal = alpha_min * feat_corr_norm + (1 - alpha_min) * feat_grad_norm 
 
        # determine Fs (sampling frequency) 
        time = np.linspace(0, pullback_len / self.speed, self.num_images) 
        self.Fs = self.num_images / np.max(time) 
        self.NFFT = int(2 ** np.ceil(np.log2(np.abs(len(self.signal))))) 
        self.sample_signal = np.fft.fft(self.signal, self.NFFT, 0) / len(self.signal) 
        self.freq = self.Fs / 2 * np.linspace(0, 1, self.NFFT // 2 + 1) 
 
        _, lower_bound_freq, upper_bound_freq = self.signal_processing() 
        # for i in range(10): 
        #     tags_dia, lower_bound_freq_1, upper_bound_freq_1 = self.signal_processing(lower_bound_freq, upper_bound_freq) 
        self.tags_dia, _, _ = self.signal_processing(lower_bound_freq, upper_bound_freq) 
         
        return self.tags_dia 
 
    def normxcorr(self, image1, image2): 
        C = np.zeros_like(image1) 
        image1_mean = np.mean(image1) 
        image2_mean = np.mean(image2) 
        image1_std = np.std(image1) 
        image2_std = np.std(image2) 
        C = np.sum((image1 - image1_mean) * (image2 - image2_mean)) / (image1_std * image2_std) 
        C = C / (image1.shape[0] * image1.shape[1]) 
        return C 
     
    def signal_processing(self, lower_bound_freq = 0.67, upper_bound_freq = 2.83): 
        # in order to return correct amplitude fft must be divided by length of sample 
        # remove non-physiological frequencies, default heart rate range is 40-170 bpm 
        self.lower_bound_freq = lower_bound_freq 
        self.upper_bound_freq = upper_bound_freq 
 
        self.freq[self.freq < lower_bound_freq] = 0 
        self.freq[self.freq > upper_bound_freq] = 0 
 
        sample_signal_red = self.sample_signal[0 : self.NFFT // 2 + 1] 
        sample_signal_red[self.freq == 0] = 0 
 
        # determine maximum frequency component of ss 
        max_freq_ss = np.argmax(np.abs(sample_signal_red)) 
        max_freq_ss = self.freq[max_freq_ss] 
 
        # find cutoff frequency 
        sigma = 0.4 
        cutoff_freq = (1 + sigma) * max_freq_ss 
 
        # construct low pass kernel (nyqu_freq is half of the transducer frame rate) Nyquist frequency max that can be represented 
        nyqu_freq = self.Fs / 2 
        tau = 25 / 46 
        v = 21 / 46 
        kernel_freq = ((cutoff_freq / nyqu_freq) * np.sinc((cutoff_freq * np.arange(1, self.num_images + 1)) / nyqu_freq)) * ( 
            tau - v * np.cos(2 * np.pi * (np.arange(1, self.num_images + 1) / self.num_images)) 
        ) 
 
        # determine low frequency signal 
        s_low = np.convolve(self.signal[:, 0], kernel_freq) 
        s_low = s_low[0 : len(self.signal)]  # take first half only 
        self.s_low = s_low

        plt.plot(self.s_low) 
        plt.show() 
 
        # find first minimum in heartbeat 
        hr_frames = int(np.round(self.Fs / max_freq_ss))  # heart rate in frames 
        idx = np.argmin(s_low[0:hr_frames]) 
        tags_dia = [] 
        tags_dia.append(idx) 
        k = 0 
 
        while idx < (self.num_images - hr_frames): 
            k = k + 1 
            # based on known heart rate seach within 2 frames for local minimum 
            # increase cutoff_freq to look at higher frequencies 
            idx2 = idx + np.arange(hr_frames - 2, hr_frames + 3) 
            # find local minimum 
            idx2 = idx2[idx2 < self.num_images - 1] 
            min_idx = np.argmin(s_low[idx2]) 
            idx = idx2[min_idx] 
            tags_dia.append(idx) 
 
        cutoff_freq = cutoff_freq + max_freq_ss 
        j = 1 
        s_low = np.expand_dims(s_low, 1) 
 
        while cutoff_freq < nyqu_freq: 
            # recalculate f with new cutoff_freq value 
            kernel_freq = ((cutoff_freq / nyqu_freq) * np.sinc((cutoff_freq * np.arange(1, self.num_images + 1)) / nyqu_freq)) * ( 
                tau - v * np.cos(2 * np.pi * (np.arange(1, self.num_images + 1) / self.num_images)) 
            ) 
            # add extra columns to s_low to create surface as in paper 
            s_temp = np.expand_dims(np.convolve(self.signal[:, 0], kernel_freq), 1) 
            s_low = np.concatenate((s_low, s_temp[0 : len(self.signal)]), 1) 
 
            # adjust each previous minimum p(i) to the nearest minimum 
            for i in range(len(tags_dia)): 
                # find index of new lowest p in +/-1 neighbour search 
                search_index = np.arange(tags_dia[i] - 1, tags_dia[i] + 2) 
                search_index = search_index[search_index >= 0] 
                search_index = search_index[search_index < len(self.signal)]  # <=? 
                search_index = np.in1d(np.arange(0, len(self.signal)), search_index) 
                # determine index of min value in the specified neighbour range 
                min_value = np.argmin(s_low[search_index, j]) 
                # switch from logical to indexed values 
                search_index = np.argwhere(search_index) 
                tags_dia[i] = search_index[min_value][0] 
            # iteratively adjust to the new minimum 
            # increase cutoff_freq to look at higher frequencies 
            cutoff_freq = cutoff_freq + max_freq_ss 
            j = j + 1 
 
        # fig = plt.figure() 
        # ax = fig.add_subplot(111, projection='3d') 
        # ax.plot_surface(*np.meshgrid(range(s_low.shape[0]), range(s_low.shape[1])), s_low.T, cmap='viridis') 
        # plt.show() 
 
        # group images between p(i) and p(i+1) 
        # output frames corresponding to each cardiac phase 
        heartbeat = [] 
        for i in range(len(tags_dia) - 1): 
            heartbeat.append(list(np.arange(tags_dia[i], tags_dia[i + 1]))) 
        # print the heart rate between two elements in the tags_dia list 
        HR_between = [] 
        for i in range(len(tags_dia) - 1): 
            HR_between.append(round(((tags_dia[i + 1] - tags_dia[i])/self.frame_rate)*60)) 
        lower_bound_freq = (np.mean(HR_between) - 2 * np.std(HR_between))/60 
        upper_bound_freq = (np.mean(HR_between) + 2 * np.std(HR_between))/60 
        print(HR_between) 
        print(tags_dia) 
        # print(lower_bound_freq) 
        # print('___') 
        # print(upper_bound_freq) 
        # print('___') 
        # print(np.mean(HR_between)) 
        return tags_dia, lower_bound_freq, upper_bound_freq 
    
    def IVUS_gating_systole(self):
        """Find the maximum signal value in between two diastolic frames and use this as the systolic frame"""
        print('IVUS_gating_systole')
        print(self.s_low)
        print(self.tags_dia)
        tags_sys = []
        distance_frames = []

        for i in range(len(self.tags_dia) - 1):
            start_idx = self.tags_dia[i] + int(0.25 * (self.tags_dia[i + 1] - self.tags_dia[i]))
            end_idx = self.tags_dia[i] + int(0.75 * (self.tags_dia[i + 1] - self.tags_dia[i]))
            
            # Find the maximum value within the specified range
            max_idx = start_idx + np.argmax(self.s_low[start_idx:end_idx])
            tags_sys.append(max_idx)
            distance_frames.append(self.tags_dia[i + 1] - max_idx)
        
        print(tags_sys)
            # Plot s_low
        plt.plot(self.s_low)
        
        # Add blue vertical lines for diastolic indices
        for dia_idx in self.tags_dia:
            plt.axvline(x=dia_idx, color='blue', linestyle='--')
        
        # Add red vertical lines for systolic indices
        for sys_idx in tags_sys:
            plt.axvline(x=sys_idx, color='red', linestyle='--')

        # Show the plot
        plt.show()

        return tags_sys, distance_frames