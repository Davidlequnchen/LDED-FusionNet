import pandas as pd
import numpy as np
import scipy as sp
import os
import math
import sys

# to make this notebook's output stable across runs
np.random.seed(42)

pd.plotting.register_matplotlib_converters()


import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

## Audio signal processing libraries
import librosa
import librosa.display
import IPython.display as ipd

import scaleogram as scg 
from glob import glob
import scipy
from scipy.signal import welch
import wave                    # library handles the parsing of WAV file headers
import pywt
import soundfile as sf
import matplotlib.font_manager as font_manager

def amplitude_envelope(signal, frame_size, hop_length):
    """Calculate the amplitude envelope of a signal with a given frame size nad hop length."""
    amplitude_envelope = []
    
    # calculate amplitude envelope for each frame
    for i in range(0, len(signal), hop_length): 
        amplitude_envelope_current_frame = max(signal[i:i+frame_size]) 
        amplitude_envelope.append(amplitude_envelope_current_frame)
    
    return np.array(amplitude_envelope) 

def mfcc_feature_normalize(data):               # data is in numpy array (40,x)
    data_normalize = []
    data_max = np.max(data, axis =0)       # max value per row, 
    data_min = np.min(data, axis =0)       # min value per row
    for i in range(len(data)):
        data_nor = [(x - data_min[i])/(data_max[i] - data_min[i]) for x in data[i]] #normalize each value in 'x'
        data_normalize.append(data_nor)
    return np.asarray(data_normalize)      #convert back to numpy array

def plot_mfcc_fixed_colorbar(Y, sr, title, frame_size=1024, hop_length = 512, y_axis="log", vmin=-80, vmax=40, cmap="magma"):
    plt.figure(figsize=(8, 5))

    # get the mfcc feature
    mfccs = librosa.feature.mfcc(y =Y, sr = sr, n_mfcc = 20)

    # Normalize mfccs to be within [0,1]
    mfccs_sc = mfcc_feature_normalize(mfccs)

    img=librosa.display.specshow(mfccs_sc, 
                                sr=sr, 
                                hop_length=hop_length, 
                                x_axis="time", 
                                y_axis=y_axis,
                                # cmap=cmap, #'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet'
                                vmin = vmin, vmax = vmax
                                )
    # plt.colorbar(img, format="%+5.f dB")
    plt.colorbar(img)
    plt.title(title, fontsize=18, pad=14)
    plt.ylabel('Frequency', fontsize = 16)
    plt.xlabel('Time', fontsize = 16)#(μs)
    
    
    
def plot_mfcc(Y, sr, title, frame_size=1024, hop_length = 512, y_axis="log", vmin=-80, vmax=40, cmap="magma"):
    plt.figure(figsize=(8, 5))

    # get the mfcc feature
    mfccs = librosa.feature.mfcc(y =Y, sr = sr, n_mfcc = 20)

    # Normalize mfccs to be within [0,1]
    # mfccs_sc = mfcc_feature_normalize(mfccs)

    img=librosa.display.specshow(mfccs_sc, 
                                sr=sr, 
                                hop_length=hop_length, 
                                x_axis="time", 
                                y_axis=y_axis,
                                cmap=cmap, #'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet'
                                vmin = vmin, vmax = vmax
                                )
    # plt.colorbar(img, format="%+5.f dB")
    plt.colorbar(img)
    plt.title(title, fontsize=18)
    plt.ylabel('Frequency', fontsize = 16)
    plt.xlabel('Time', fontsize = 16)  #(μs)

def plot_spectrogram(Y, sr, title, frame_size=1024, hop_length = 512, y_axis="log", vmin=-80, vmax=40, cmap="magma"):
    plt.figure(figsize=(8, 5))
    
    Stft = librosa.stft(Y, n_fft=frame_size, hop_length=hop_length)
    Y_sample  = np.abs(Stft) ** 2
    Y_log_sample = librosa.power_to_db(Y_sample)

    img=librosa.display.specshow(Y_log_sample, 
                                sr=sr, 
                                hop_length=hop_length, 
                                x_axis="time", 
                                y_axis=y_axis,
                                # cmap=cmap, #'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet'
                                vmin = vmin, vmax = vmax
                                )
    plt.colorbar(img, format="%+2.f dB")
    plt.title(title, fontsize=18, pad=1.6)
    plt.ylabel('Frequency', fontsize = 16)
    plt.xlabel('Time', fontsize = 16)  #(μs)

    
def fft_signal_visualization_smooth(signal_list, name_list, f_ratio=0.5, log_x=False, log_y=False, sr=44100, N_smooth = 1000):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(8, 4)) #constrained_layout=True,
    
    axs.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    for i in range(len(signal_list)):
        signal_data = signal_list[i]
        X_mag = np.absolute(np.fft.fft(signal_data))
        # freqeuncy bands
        f_bands = np.linspace(0, sr, len(X_mag))
        f_bins = int(len(X_mag)*f_ratio)
        
        X_mag_mean = pd.Series(X_mag[:f_bins]).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        plt.plot( f_bands[:f_bins-N_smooth+1], X_mag_mean, alpha=1, label=name_list[i])
    
    # ax.set_ylim(0, 10000)
    if log_x and not log_y:
        axs.set_xscale('log') 
        plt.xlabel('Frequency (Hz)', fontsize=16)
        plt.ylabel('Magnitude', fontsize=16)
        # plt.title("Log scale freqeuncy comparison", fontsize=18)
    if log_y and not log_x:
        axs.set_yscale('log')
        plt.xlabel('Frequency (Hz)', fontsize=16)
        plt.ylabel('Magnitude (db)', fontsize=16)
        # plt.title("Log scale magnitude comparison", fontsize=18)
    else:
        plt.xlabel('Frequency (Hz)', fontsize=16)
        plt.ylabel('Magnitude', fontsize=16)
        # plt.title("FFT plots", fontsize=18)
    
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=12)
    # axs.legend(loc = 3) 
    axs.legend(loc="best", prop=font)   
    
    
    
def equalized_signal_visualization(noisy, cleaned, sr=44100, alpha = 1):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(8, 4)) #constrained_layout=True,

    librosa.display.waveplot(noisy, sr=sr, alpha=alpha, label = 'original signal')
    librosa.display.waveplot(cleaned,sr=sr, alpha=0.6, label = 'eqaulized signal')
    axs.set_xlabel('Time', fontsize = 16)
    axs.set_ylabel('Normalized Amplitute', fontsize = 16)
    
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=14)
    axs.legend(loc = 3, prop=font)
    # fig.suptitle("Time-domain visualisation", fontsize = 20,  y=1)


def final_signal_visualization(filtered, final, sr=44100, alpha = 1):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(8, 4)) #constrained_layout=True,

    librosa.display.waveplot(filtered, sr=sr, alpha=alpha, label = 'raw signal')
    librosa.display.waveplot(final,sr=sr, alpha=0.5, label = 'clean signal')
    axs.set_xlabel('Time', fontsize = 16)
    axs.set_ylabel('Normalized Amplitute', fontsize = 16)

    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=14)
    axs.legend(loc = 3, prop=font)
    # fig.suptitle("Time-domain visualisation", fontsize = 20,  y=1)
    
    
def final_signal_visualization_smooth(original, final_cleaned, sr=44100, alpha = 1,  N = 100):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(8, 5)) #constrained_layout=True,

    original = pd.Series(original).rolling(window=N).mean().iloc[N-1:].values

    final_cleaned = pd.Series(final_cleaned).rolling(window=N).mean().iloc[N-1:].values
    
    
    librosa.display.waveplot(original, sr=sr, alpha=alpha, label = 'raw signal')
    librosa.display.waveplot(final_cleaned,sr=sr, alpha=0.9, label = 'denoised signal')
    axs.set_xlabel('Time', fontsize = 16)
    axs.set_ylabel('Normalized Amplitute', fontsize = 16)
    # axs.set_ylim([-1, 1])
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=14)
    axs.legend(loc = 3, prop=font)
    # fig.suptitle("Time-domain visualisation", fontsize = 20,  y=1)
    

def three_step_cleaned_signal_visualization_fft_smooth(original, equalized, bandpassed, final_cleaned, 
                                                f_ratio=0.5, log_x=False, log_y=False, sr=44100, N_smooth = 1000):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(10, 4)) #constrained_layout=True,
    axs.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    
    X_original_mag = np.absolute(np.fft.fft(original))
    X_equalized_mag = np.absolute(np.fft.fft(equalized))
    X_bandpassed_mag = np.absolute(np.fft.fft(bandpassed))
    X_final_mag = np.absolute(np.fft.fft(final_cleaned))
    
    # freqeuncy bands
    f_original = np.linspace(0, sr, len(X_original_mag))
    f_bins_original = int(len(X_original_mag)*f_ratio)  

    f_equalized = np.linspace(0, sr, len(X_equalized_mag))
    f_bins_equalized = int(len(X_equalized_mag)*f_ratio) 
    
    f_bandpassed = np.linspace(0, sr, len(X_bandpassed_mag))
    f_bins_bandpassed = int(len(X_bandpassed_mag)*f_ratio) 

    f_final = np.linspace(0, sr, len(X_final_mag))
    f_bins_final = int(len(X_final_mag)*f_ratio) 
    
    ## get profile
    # high_idx, low_idx = hl_envelopes_idx(X_original_mag)
    # plt.plot(f_original[high_idx], X_original_mag[high_idx], alpha=1, label = 'original signal')
    # plt.plot(f_original[low_idx], X_original_mag[low_idx], alpha=1, label = 'original signal')
    # plt.plot(f_original, X_original_mag, alpha=1, label = 'original signal')
    
    N = N_smooth
    X_original_mag_mean = pd.Series(X_original_mag[:f_bins_original]).rolling(window=N).mean().iloc[N-1:].values
    plt.plot( f_original[:f_bins_original-N+1], X_original_mag_mean, alpha=1, label = 'Raw signal')
    
    X_equalized_mag_mean = pd.Series(X_equalized_mag[:f_bins_equalized]).rolling(window=N).mean().iloc[N-1:].values
    plt.plot( f_equalized[:f_bins_equalized-N+1], X_equalized_mag_mean, alpha=1, label = 'Equalized (step 1)')
    
    X_bandpassed_mag_mean = pd.Series(X_bandpassed_mag[:f_bins_bandpassed]).rolling(window=N).mean().iloc[N-1:].values
    plt.plot( f_bandpassed[:f_bins_bandpassed-N+1], X_bandpassed_mag_mean, alpha=1, label = 'Bandpassed (step 2)')
    
    X_final_mag_mean = pd.Series(X_final_mag[:f_bins_final]).rolling(window=N).mean().iloc[N-1:].values
    plt.plot( f_final[:f_bins_final-N+1], X_final_mag_mean, alpha=1, label = 'Final denoised (step 3)')
 
    
    # ax.set_ylim(0, 10000)
    if log_x and not log_y:
        axs.set_xscale('log') 
        plt.xlabel('Frequency (Hz)', fontsize=16)
        plt.ylabel('Magnitude', fontsize=16)
        # plt.title("Log scale freqeuncy comparison", fontsize=18)
    if log_y and not log_x:
        axs.set_yscale('log')
        plt.xlabel('Frequency (Hz)', fontsize=16)
        plt.ylabel('Magnitude (db)', fontsize=16)
        # plt.title("Log scale magnitude comparison", fontsize=18)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=14)
    axs.legend(loc = "best", prop=font)

    
    
    
def three_step_cleaned_signal_visualization_smooth(original, equalized, bandpassed, final_cleaned, sr=44100, N = 100):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(8, 5)) #constrained_layout=True,
    
    original = pd.Series(original).rolling(window=N).mean().iloc[N-1:].values
    equalized = pd.Series(equalized).rolling(window=N).mean().iloc[N-1:].values
    bandpassed = pd.Series(bandpassed).rolling(window=N).mean().iloc[N-1:].values
    final_cleaned = pd.Series(final_cleaned).rolling(window=N).mean().iloc[N-1:].values
    
    
    librosa.display.waveplot(original, sr=sr, alpha=1, label = 'raw signal')
    librosa.display.waveplot(equalized, sr=sr, alpha=0.85, label = 'after equalization (step 1)')
    librosa.display.waveplot(bandpassed,sr=sr, alpha=0.7, label = 'after bandpass filter (step 2)')
    librosa.display.waveplot(final_cleaned,sr=sr, alpha=0.5, label = 'final clean audio (step 3)')
    axs.set_xlabel('Time', fontsize = 16)
    axs.set_ylabel('Normalized Amplitute', fontsize = 16)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=14)
    axs.legend(loc = 3, ncol=2, prop=font) #handleheight=2.4, labelspacing=0.05
    # fig.suptitle("Time-domain visualisation", fontsize = 20,  y=1)
    
def category_visualization_smooth(signal_1, signal_2, signal_3, signal_4, sr=44100, N = 100):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(8, 5)) #constrained_layout=True,
    
    signal_1 = pd.Series(signal_1).rolling(window=N).mean().iloc[N-1:].values
    signal_2 = pd.Series(signal_2).rolling(window=N).mean().iloc[N-1:].values
    signal_3 = pd.Series(signal_3).rolling(window=N).mean().iloc[N-1:].values
    signal_4 = pd.Series(signal_4).rolling(window=N).mean().iloc[N-1:].values
    
    
    librosa.display.waveplot(signal_4, sr=sr, alpha=1, label = 'Extremely low efficiency')
    librosa.display.waveplot(signal_1, sr=sr, alpha=0.8, label = 'Low efficiency')
    librosa.display.waveplot(signal_2,sr=sr, alpha=0.8, label = 'Medium efficiency')
    librosa.display.waveplot(signal_3,sr=sr, alpha=0.7, label = 'High efficiency')
    axs.set_xlabel('Time', fontsize = 16)
    axs.set_ylabel('Normalized Amplitute', fontsize = 16)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=14)
    axs.legend(loc = 3, ncol=2, prop=font) #handleheight=2.4, labelspacing=0.05
    # fig.suptitle("Time-domain visualisation", fontsize = 20,  y=1)
    
def visualize_rms_smooth(signal_list, name_list, frame_size, hop_length, sampling_rate, N_smooth=100):
    plt.figure(figsize=(8, 5))
    for i in range(len(signal_list)):
        signal_data = signal_list[i]
        rms= librosa.feature.rms(signal_data, frame_length=frame_size, hop_length=hop_length)[0]
        rms_smooth = pd.Series(rms).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        rms_smooth_further = pd.Series(rms).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add
   
        ## visualisation
        t = librosa.frames_to_time(range(len(rms_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(rms_smooth_further)), hop_length=hop_length, sr = sampling_rate) #######add
        plt.plot(t, rms_smooth, label=name_list[i])
        plt.plot(t2, rms_smooth_further, label=name_list[i]+" smooth", c='orangered')#########add
        
        plt.ylabel('RMS Energy', fontsize = 16)
        plt.xlabel('Time (s)', fontsize = 16)  #(μs)
    # plt.title("RMS Energy", fontsize = 18, pad=15)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font) #'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'

def visualize_envelope_smooth(signal_list, name_list, frame_size, hop_length, sampling_rate, N_smooth=100):
    plt.figure(figsize=(8, 5))
    for i in range(len(signal_list)):
        signal_data = signal_list[i]
        ae = amplitude_envelope(signal_data, frame_size, hop_length)
        ae_smooth = pd.Series(ae).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        ae_smooth_further = pd.Series(ae).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add
   
        ## visualisation
        t = librosa.frames_to_time(range(len(ae_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(ae_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, ae_smooth, label=name_list[i])
        plt.plot(t2, ae_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add
        
        plt.ylabel('Amplitute', fontsize = 16)
        plt.xlabel('Time (s)', fontsize = 16)  #(μs)
    # plt.title("Amplitude envolope", fontsize = 18, pad=15)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="upper right", prop=font)


def spectral_centroid(signal_list,  name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth = 100):
    plt.figure(figsize=(8, 5))
    for i in range(len(signal_list)):
        signal_data = signal_list[i]
        sc = librosa.feature.spectral_centroid(y=signal_data, sr=sampling_rate, n_fft=frame_size, hop_length=hop_length)[0]
        sc_smooth = pd.Series(sc).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        sc_smooth_further = pd.Series(sc).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add

        t = librosa.frames_to_time(range(len(sc_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(sc_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, sc_smooth, label=name_list[i])
        plt.plot(t2, sc_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add

    plt.ylabel('Frequency (Hz)', fontsize = 16)
    plt.xlabel('Time (s)', fontsize = 16)  #(μs)
    # plt.title("Spectrial centroid", fontsize = 18)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)


def spectral_bandwidth(signal_list, name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth = 100, power=2):
    plt.figure(figsize=(8, 5))
    for i in range(len(signal_list)):
        signal_data = signal_list[i]
        sb = librosa.feature.spectral_bandwidth(y=signal_data, sr=sampling_rate, n_fft=frame_size, hop_length=hop_length, p=power)[0]
        sb_smooth = pd.Series(sb).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        sb_smooth_further = pd.Series(sb).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add

        t = librosa.frames_to_time(range(len(sb_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(sb_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, sb_smooth, label=name_list[i])
        plt.plot(t2, sb_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add

    plt.ylabel('Frequency (Hz)', fontsize = 16)
    plt.xlabel('Time (s)', fontsize = 16)  #(μs)
    # plt.title("Spectrial bandwidth", fontsize = 18)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)

    
def spectral_rolloff(signal_list, name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth = 100, roll_percent=0.92):
    plt.figure(figsize=(8, 5))
    for i in range(len(signal_list)):
        signal_data = signal_list[i]
        # Approximate maximum frequencies with roll_percent=0.99
        rolloff = librosa.feature.spectral_rolloff(y=signal_data, sr=sampling_rate, roll_percent=roll_percent)
        rolloff_smooth = pd.Series(rolloff[0]).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        rolloff_smooth_further = pd.Series(rolloff[0]).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add

        t = librosa.frames_to_time(range(len(rolloff_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(rolloff_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        
        plt.plot(t, rolloff_smooth, label=name_list[i])
        plt.plot(t2, rolloff_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add

    plt.ylabel('Frequency (Hz)', fontsize = 16)
    plt.xlabel('Time (s)', fontsize = 16)  #(μs)
    # plt.title('Spectral Rolloff at {:2.2%} freqeuncy'.format(roll_percent), fontsize = 18, y=1.0005)
    
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)

    
def spectral_flatness(signal_list, name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth = 100, power=2):
    plt.figure(figsize=(8, 5))
    for i in range(len(signal_list)):
        signal_data = signal_list[i]
        
        # S, phase = librosa.magphase(librosa.stft(signal_data))
        # if power:
        #     S_power = S ** 2
        # else:
        #     S_power = S
        
        spectral_flatness = librosa.feature.spectral_flatness(signal_data, power=power, n_fft=frame_size, hop_length=hop_length)[0]
        spectral_flatness_smooth = pd.Series(spectral_flatness).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values

        spectral_flatness_smooth_further = pd.Series(spectral_flatness).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add
        t = librosa.frames_to_time(range(len(spectral_flatness_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(spectral_flatness_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, spectral_flatness_smooth, label=name_list[i])
        plt.plot(t2, spectral_flatness_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add
    
    plt.ylabel('Spectral Flatness', fontsize = 16)
    plt.xlabel('Time (s)', fontsize = 16)  #(μs)
    # plt.title('Spectral Flatness', fontsize = 18, y=1.0005)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)
    
def calculate_split_frequency_bin(split_frequency, sample_rate, num_frequency_bins):
    """Infer the frequency bin associated to a given split frequency."""
    frequency_range = sample_rate / 2
    frequency_delta_per_bin = frequency_range / num_frequency_bins
    split_frequency_bin = math.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)


def band_energy_ratio(spectrogram, split_frequency, sample_rate):
    """Calculate band energy ratio with a given split frequency."""
    split_frequency_bin = calculate_split_frequency_bin(split_frequency, sample_rate, len(spectrogram[0]))
    band_energy_ratio = []
    
    # calculate power spectrogram
    # power_spectrogram = np.abs(spectrogram) ** 2
    power_spectrogram = np.abs(spectrogram) 
    power_spectrogram = power_spectrogram.T
    
    # calculate BER value for each frame
    for frame in power_spectrogram:
        sum_power_low_frequencies = frame[:split_frequency_bin].sum()
        sum_power_high_frequencies = frame[split_frequency_bin:].sum()
        band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(band_energy_ratio_current_frame)
    
    return np.array(band_energy_ratio)
    
def band_energy_ratio_plot(signal_list, name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth = 100, split_frequency = 7500):
    plt.figure(figsize=(8, 5))
    for i in range(len(signal_list)):
        signal_data = signal_list[i]
    
        Stft = librosa.stft(signal_data, n_fft=frame_size, hop_length=hop_length)
        ber = band_energy_ratio(Stft, split_frequency = split_frequency, sample_rate = sampling_rate)
        ber_smooth = pd.Series(ber).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        ber_smooth_further = pd.Series(ber).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add

        t = librosa.frames_to_time(range(len(ber_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(ber_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, ber_smooth,  label=name_list[i])
        plt.plot(t2, ber_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add

    plt.ylabel('BER', fontsize = 16)
    plt.xlabel('Time (s)', fontsize = 16)  #(μs)
    # plt.title('Band Energy Ratio', fontsize = 18, y=1.0005)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)
    
def spectral_contrast(signal_list, name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth = 100, n_bands=6, quantile=0.02, linear=False):
    plt.figure(figsize=(8, 5))
    for i in range(len(signal_list)):
        signal_data = signal_list[i]
        spectral_contrast = librosa.feature.spectral_contrast(y=signal_data, sr=sampling_rate, n_fft=frame_size, 
                                                              hop_length=hop_length, n_bands=n_bands, quantile=quantile, linear=linear)[0]
        spectral_contrast_smooth = pd.Series(spectral_contrast).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        spectral_contrast_smooth_further = pd.Series(spectral_contrast).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add
        t = librosa.frames_to_time(range(len(spectral_contrast_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(spectral_contrast_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, spectral_contrast_smooth,  label=name_list[i])
        plt.plot(t2, spectral_contrast_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add

    plt.ylabel('Spectrial Contrast', fontsize = 16)
    plt.xlabel('Time (s)', fontsize = 16)  #(μs)
    # plt.title("Spectrial Contrast", fontsize = 18)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)

def energy_entropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    eps = 0.000000000001
    # total frame energy
    frame_energy = np.sum(frame ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy


def plot_energy_entropy(signal_list, name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth=200):
    plt.figure(figsize=(8, 5))
    for i in range(len(signal_list)):
        signal_data = signal_list[i] 
        number_of_samples = len(signal_data)  # total number of samples
        current_position = 0
        count_fr = 0
        num_fft = int(frame_size / 2)
        
        e_entropy_features = []
        # for each short-term window to end of signal
        while current_position + frame_size - 1 < number_of_samples:
            count_fr += 1
            # get current window
            x = signal_data[current_position:current_position + frame_size]

            # update window position
            current_position = current_position + hop_length


            e_entropy = energy_entropy(x, n_short_blocks=10)
            e_entropy_features.append(e_entropy)
            
        # s_entropy_features = np.concatenate(s_entropy, 1)
        # print (len(s_entropy))
        e_entropy_smooth = pd.Series(e_entropy_features).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        e_entropy_smooth_further = pd.Series(e_entropy_features).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add
        t = librosa.frames_to_time(range(len(e_entropy_smooth)), hop_length=hop_length, sr = 44100)
        t2 = librosa.frames_to_time(range(len(e_entropy_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, e_entropy_smooth, label=name_list[i])
        plt.plot(t2, e_entropy_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add
      
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)
    
    
    
def zero_crossing_rate_plot(signal_list, name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth = 100):
    plt.figure(figsize=(8, 5))
    for i in range(len(signal_list)):
        signal_data = signal_list[i]

        zcr= librosa.feature.zero_crossing_rate(signal_data, frame_length=frame_size, hop_length=hop_length)[0]
        zcr_smooth = pd.Series(zcr).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        zcr_smooth_further = pd.Series(zcr).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add
        ## visualisation
        t = librosa.frames_to_time(range(len(zcr_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(zcr_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, zcr_smooth, label=name_list[i])
        plt.plot(t2, zcr_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add
    
    plt.ylabel('ZCR', fontsize = 16)
    plt.xlabel('Time', fontsize = 16)  #(μs)
    # plt.title('Zero Crossing Rate', fontsize = 18, y=1.0005)
    
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)
    
    
#-----------------------------------------------------------------------
def spectral_crest(spectrum):
    """
    Compute the spectral crest factor, i.e. the ratio of the maximum of the spectrum to the
    sum of the spectrum
    reference:
    - https://github.com/jsawruk/pymir/blob/master/pymir/Spectrum.py
    - https://dsp.stackexchange.com/questions/27221/calculating-rms-crest-factor-for-a-stereo-signal
    """
    # absSpectrum = abs(spectrum)
    absSpectrum = np.absolute(spectrum)
    spectralSum = np.sum(absSpectrum)

    maxFrequencyIndex = np.argmax(absSpectrum)
    # maxSpectrum = absSpectrum[maxFrequencyIndex]
    maxSpectrum = np.amax(absSpectrum, axis=0)

    return maxSpectrum / spectralSum


def spectral_skewness(spectrum):
    """
    Compute the spectral skewness (third spectral moment)
    """
    return scipy.stats.skew(abs(spectrum))


def spectral_variance(spectrum):
    """
    Compute the spectral variance (second spectral moment)
    """
    # return np.var(abs(spectrum))
    return scipy.stats.variation(abs(spectrum))

def spectral_kurtosis(spectrum):
    """
    Compute the spectral kurtosis (fourth spectral moment)
    """
    return scipy.stats.kurtosis(abs(spectrum))


def plot_spectral_kurtosis(signal_list, name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth=200):
    plt.figure(figsize=(8, 5))
    
    for i in range(len(signal_list)):
        signal_data = signal_list[i]    
        S, phase = librosa.magphase(librosa.stft(signal_data))
        # S_power = S **2   
        S_power = S
        sk = spectral_kurtosis(S_power)
        sk_smooth = pd.Series(sk).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        sk_smooth_further = pd.Series(sk).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add

        t = librosa.frames_to_time(range(len(sk_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(sk_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, sk_smooth, label=name_list[i])
        plt.plot(t2, sk_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add
      
    # plt.title("Spectral Kurtosis", fontsize = 16)
    plt.ylabel('Spectral kurtosis', fontsize = 14)
    plt.xlabel('Time (s)', fontsize = 14)  #(μs)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)
    
def plot_spectral_variance(signal_list, name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth=200):
    plt.figure(figsize=(8, 5))
    
    for i in range(len(signal_list)):
        signal_data = signal_list[i]    
        S, phase = librosa.magphase(librosa.stft(signal_data))
        # S_power = S **2   
        S_power = S
        sv = spectral_variance(S_power)
        sv_smooth = pd.Series(sv).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        sv_smooth_further = pd.Series(sv).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add
        t = librosa.frames_to_time(range(len(sv_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(sv_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, sv_smooth, label=name_list[i])
        plt.plot(t2, sv_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add
      
    # plt.title("Spectral Variance", fontsize = 16)
    plt.ylabel('Spectral variance', fontsize = 14)
    plt.xlabel('Time (s)', fontsize = 14)  #(μs)
    # plt.ylim(0,0.04)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)
    
        
def plot_spectral_crest(signal_list, name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth=200):
    plt.figure(figsize=(8, 5))
    
    for i in range(len(signal_list)):
        signal_data = signal_list[i]    
        S, phase = librosa.magphase(librosa.stft(signal_data))
        # S_power = S **2   
        S_power = S
        sc = spectral_crest(S_power)
        sc_smooth = pd.Series(sc).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        sc_smooth_further = pd.Series(sc).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add
        t = librosa.frames_to_time(range(len(sc_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(sc_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, sc_smooth, label=name_list[i])
        plt.plot(t2, sc_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add
      
    # plt.title("Spectral Crest", fontsize = 16)
    plt.ylabel('Spectral crest', fontsize = 14)
    plt.xlabel('Time (s)', fontsize = 14)  #(μs)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)
    
    
def plot_spectral_skewness(signal_list,name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth=200):
    plt.figure(figsize=(8, 5))
    
    for i in range(len(signal_list)):
        signal_data = signal_list[i]    
        S, phase = librosa.magphase(librosa.stft(signal_data))
        # S_power = S **2   
        S_power = S
        s_skewness = spectral_skewness(S_power)
        s_skewness_smooth = pd.Series(s_skewness).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        s_skewness_smooth_further = pd.Series(s_skewness).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add
        t = librosa.frames_to_time(range(len(s_skewness_smooth)), hop_length=hop_length, sr = sampling_rate)
        t2 = librosa.frames_to_time(range(len(s_skewness_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, s_skewness_smooth, label=name_list[i])
        plt.plot(t2, s_skewness_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add
      
    # plt.title("Spectral Skewness", fontsize = 16)
    plt.ylabel('Spectral skewness', fontsize = 14)
    plt.xlabel('Time (s)', fontsize = 14)  #(μs)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)

    
def spectral_flux(fft_magnitude, previous_fft_magnitude):
    eps = 0.00000000000000001
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        fft_magnitude:            the abs(fft) of the current frame
        previous_fft_magnitude:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    fft_sum = np.sum(fft_magnitude + eps)
    previous_fft_sum = np.sum(previous_fft_magnitude + eps)
    sp_flux = np.sum(
        (fft_magnitude / fft_sum - previous_fft_magnitude /
         previous_fft_sum) ** 2)

    return sp_flux

def spectral_entropy(signal, n_short_blocks=10):
    eps = 0.00000001
    """Computes the spectral entropy"""
    # number of frame samples
    num_frames = len(signal)

    # total spectral energy
    total_energy = np.sum(signal ** 2)

    # length of sub-frame
    sub_win_len = int(np.floor(num_frames / n_short_blocks))
    if num_frames != sub_win_len * n_short_blocks:
        signal = signal[0:sub_win_len * n_short_blocks]

    # define sub-frames (using matrix reshape)
    sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # compute spectral sub-energies
    s = np.sum(sub_wins ** 2, axis=0) / (total_energy + eps)

    # compute spectral entropy
    entropy = -np.sum(s * np.log2(s + eps))

    return entropy


def plot_spectral_entropy(signal_list, name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth=200):
    plt.figure(figsize=(8, 5))
    for i in range(len(signal_list)):
        signal_data = signal_list[i] 
        number_of_samples = len(signal_data)  # total number of samples
        current_position = 0
        count_fr = 0
        num_fft = int(frame_size / 2)
        
        s_entropy_features = []
         # for each short-term window to end of signal
        while current_position + frame_size - 1 < number_of_samples:
            count_fr += 1
            # get current window
            x = signal_data[current_position:current_position + frame_size]

            # update window position
            current_position = current_position + hop_length

            # get fft magnitude
            fft_magnitude = abs(fft(x))

            # normalize fft
            fft_magnitude = fft_magnitude[0:num_fft]
            fft_magnitude = fft_magnitude / len(fft_magnitude)

            s_entropy = spectral_entropy(fft_magnitude, n_short_blocks=10)
            s_entropy_features.append(s_entropy)
            
        # s_entropy_features = np.concatenate(s_entropy, 1)
        # print (len(s_entropy))
        s_entropy_smooth = pd.Series(s_entropy_features).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        s_entropy_smooth_smooth_further = pd.Series(s_entropy_features).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add
        t = librosa.frames_to_time(range(len(s_entropy_smooth)), hop_length=hop_length, sr = 44100)
        t2 = librosa.frames_to_time(range(len(s_entropy_smooth_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, s_entropy_smooth,  label=name_list[i])
        plt.plot(t2, s_entropy_smooth_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add
      
    plt.title("Spectral Entropy", fontsize = 16)
    plt.ylabel('Entropy', fontsize = 14)
    plt.xlabel('Time (s)', fontsize = 14)  #(μs)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)
    
    
    
def plot_spectral_flux(signal_list, name_list, frame_size, hop_length, sampling_rate = 44100, N_smooth=200):
    plt.figure(figsize=(8, 5))
    for i in range(len(signal_list)):
        signal_data = signal_list[i] 
        number_of_samples = len(signal_data)  # total number of samples
        current_position = 0
        count_fr = 0
        num_fft = int(frame_size / 2)
        
        s_flux_features = []
         # for each short-term window to end of signal
        while current_position + frame_size - 1 < number_of_samples:
            count_fr += 1
            # get current window
            x = signal_data[current_position:current_position + frame_size]

            # update window position
            current_position = current_position + hop_length

            # get fft magnitude
            fft_magnitude = abs(fft(x))

            # normalize fft
            fft_magnitude = fft_magnitude[0:num_fft]
            fft_magnitude = fft_magnitude / len(fft_magnitude)
            
            # keep previous fft mag (used in spectral flux)
            if count_fr == 1:
                fft_magnitude_previous = fft_magnitude.copy()

            s_flux = spectral_flux(fft_magnitude,fft_magnitude_previous)
            s_flux_features.append(s_flux)
            
            fft_magnitude_previous = fft_magnitude.copy()
            
        # s_entropy_features = np.concatenate(s_entropy, 1)
        # print (len(s_entropy))
        s_flux_smooth = pd.Series(s_flux_features).rolling(window=N_smooth).mean().iloc[N_smooth-1:].values
        s_flux_smooth_smooth_further = pd.Series(s_flux_smooth).rolling(window=N_smooth+400).mean().iloc[N_smooth+400-1:].values #########add
        t = librosa.frames_to_time(range(len(s_flux_smooth)), hop_length=hop_length, sr = 44100)
        t2 = librosa.frames_to_time(range(len(s_flux_smooth_smooth_further)), hop_length=hop_length, sr = sampling_rate) #########add
        plt.plot(t, s_flux_smooth, label=name_list[i])
        plt.plot(t2, s_flux_smooth_smooth_further, label=name_list[i]+" smooth", c='orangered') #########add
      
    # plt.title("Spectral Flux", fontsize = 16)
    plt.ylabel('Spectral Flux', fontsize = 14)
    plt.xlabel('Time (s)', fontsize = 14)  #(μs)
    font = font_manager.FontProperties(family='sans-serif',
                                   # weight='bold',
                                   style='normal', size=13)
    plt.grid(linestyle='--', linewidth=1, alpha=0.7) #linestyle='--', color='r'
    plt.legend(loc="best", prop=font)


# from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# def spectral_entropy(spectrum, frame_size,  method='fft', normalize=False):
#     return ant.spectral_entropy(spectrum, sf=44100, method=method, nperseg=frame_size, normalize=normalize)
    
    
def feature_extraction_audioAnalysis(signal, Fs=44100):
    F, f_names = ShortTermFeatures.feature_extraction(signal, Fs, 0.050*Fs, 0.025*Fs)
    plt.plot(F[5,:]); plt.xlabel('Frame number'); plt.ylabel(f_names[5]) 
    return F[5,:]

def spectral_entropy(signal, n_short_blocks=10):
    eps = 0.00000001
    """Computes the spectral entropy"""
    # number of frame samples
    num_frames = len(signal)

    # total spectral energy
    total_energy = np.sum(signal ** 2)

    # length of sub-frame
    sub_win_len = int(np.floor(num_frames / n_short_blocks))
    if num_frames != sub_win_len * n_short_blocks:
        signal = signal[0:sub_win_len * n_short_blocks]

    # define sub-frames (using matrix reshape)
    sub_wins = signal.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # compute spectral sub-energies
    s = np.sum(sub_wins ** 2, axis=0) / (total_energy + eps)

    # compute spectral entropy
    entropy = -np.sum(s * np.log2(s + eps))

    return entropy


def spectral_flux(fft_magnitude, previous_fft_magnitude):
    eps = 0.00000000000000001
    """
    Computes the spectral flux feature of the current frame
    ARGUMENTS:
        fft_magnitude:            the abs(fft) of the current frame
        previous_fft_magnitude:        the abs(fft) of the previous frame
    """
    # compute the spectral flux as the sum of square distances:
    fft_sum = np.sum(fft_magnitude + eps)
    previous_fft_sum = np.sum(previous_fft_magnitude + eps)
    sp_flux = np.sum(
        (fft_magnitude / fft_sum - previous_fft_magnitude /
         previous_fft_sum) ** 2)

    return sp_flux

def energy_entropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    eps = 0.000000000001
    # total frame energy
    frame_energy = np.sum(frame ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy
