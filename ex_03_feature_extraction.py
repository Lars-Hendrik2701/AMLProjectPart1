import numpy as np
import pandas as pd
from scipy.signal import detrend, windows

"""
Task Description 3

In this exercise, you will implement feature extraction techniques for time-series welding data and create visualizations to analyze these features. You'll extract statistical and frequency-domain features from the welding signals and create plots to explore relationships between these features.

Objectives:

    Implement functions to extract meaningful features from time-series welding data
    Calculate statistical measures from voltage and current signals
    Extract frequency-domain features using spectral analysis
    Visualize feature relationships to gain insights for classification tasks
    Document your feature selection strategy

Tasks:
Exercise 3.1: Dominant Frequency Extraction Function

    Implement the find_dominant_frequencies(x: np.ndarray, fs: int) function in ex_03_feature_extraction.py that:
        Takes multiple input signals and a sampling frequency
        Applies detrending and windowing to prepare signals for frequency analysis
        Computes Fast Fourier Transform (FFT) to convert signals to frequency domain
        Calculates Power Spectral Density (PSD)
        Identifies and returns the dominant frequency for each signal

Exercise 3.2: Feature Extraction Function

    Implement the extract_features(data: np.ndarray, labels: np.ndarray) function that:
        Extracts 20 different features from the raw voltage and current signals.
        Features include:
          - Statistical: mean, std, median, min, max, range, RMS for voltage and current
          - Frequency: dominant frequency for voltage and current
          - Power-related: mean power, std power, max power, median power
        Returns a DataFrame with all features and quality labels

Notes:

    Use the get_welding_data() function from Exercise 1 to load the raw data
    Document why certain features might be more valuable than others for classification
    Ensure all features are scaled appropriately for visualization

This exercise builds on the data loading foundations established in Exercise 1 and prepares you for classification and clustering tasks.
"""

def find_dominant_frequencies(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Calculates the dominant frequencies of multiple input signals with FFT-based analysis.

    Args:
        x (np.ndarray): The input signals, shape: (num_samples, seq_len).
        fs (int): The sampling frequency of the signals.

    Returns:
        np.ndarray: The dominant frequency for each signal, shape: (num_samples,).
    """
    num_samples, seq_len = x.shape
    dominant_freqs = np.zeros(num_samples)
    freqs = np.fft.rfftfreq(seq_len, d=1/fs)
    window = windows.hann(seq_len) 

    for i in range(num_samples):
        signal = detrend(x[i]) #Detrending
        windowed = signal * window #windowing
        fft_vals = np.fft.rfft(windowed) #Fast Fourier Transform
        psd = np.abs(fft_vals)**2 #Power Spectral Density
        idx = np.argmax(psd[1:]) + 1 
        dominant_freqs[i] = freqs[idx]

    return dominant_freqs


def extract_features(
    data: np.ndarray,
    labels: np.ndarray,
    fs: int = 1
) -> pd.DataFrame:
    """
    Extracts a set of statistical, frequency-domain, and power-related features from welding data.

    Args:
        data (np.ndarray): Array of shape (num_samples, seq_len, 2) with currents and voltages.
        labels (np.ndarray): Array of shape (num_samples,) with quality labels.
        fs (int): Sampling frequency for FFT, default=1 (normalized).

    Returns:
        pd.DataFrame: DataFrame of shape (num_samples, num_features+1) including 'label'.
    """
    num_samples, seq_len, feat_dim = data.shape
    # Separieren der Kanäle: 0=current, 1=voltage
    current = data[:, :, 0]
    voltage = data[:, :, 1]

    features = {
        'label': labels
    }

    # Statistik
    for name, arr in [('current', current), ('voltage', voltage)]:
        features[f'{name}_mean'] = arr.mean(axis=1)
        features[f'{name}_std'] = arr.std(axis=1, ddof=1)
        features[f'{name}_median'] = np.median(arr, axis=1)
        features[f'{name}_min'] = arr.min(axis=1)
        features[f'{name}_max'] = arr.max(axis=1)
        features[f'{name}_range'] = features[f'{name}_max'] - features[f'{name}_min']
        features[f'{name}_rms'] = np.sqrt((arr**2).mean(axis=1))

    # Bestimmen der dominanten Frequenzen
    features['current_dom_freq'] = find_dominant_frequencies(current, fs)
    features['voltage_dom_freq'] = find_dominant_frequencies(voltage, fs)

    # Leistungsstatistik
    power = voltage * current
    features['power_mean'] = power.mean(axis=1)
    features['power_std'] = power.std(axis=1, ddof=1)
    features['power_max'] = power.max(axis=1)
    features['power_median'] = np.median(power, axis=1)

    # Build DataFrame
    df = pd.DataFrame(features)
    return df
