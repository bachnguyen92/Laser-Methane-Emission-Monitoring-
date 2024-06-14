import numpy as np
import scipy
import pandas as pd
import statistics
import librosa
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error
# from docx import Document
from sklearn.preprocessing import MinMaxScaler
import peakutils as pu


def init_funtion(up_limit=300, poles=9, delta=50, Fs=6.3e6, f=10000):
    delta = delta / 2
    bp_sos = scipy.signal.butter(poles, [f - delta, f + delta], 'bandpass', fs=Fs, analog=False, output='sos')

    mb_sos = []

    hp_sos = scipy.signal.butter(poles, 2 * f - delta, 'high', fs=Fs, analog=False, output='sos')
    mb_sos.append(hp_sos)

    for cfreq in range(2, up_limit, 1):
        sos = scipy.signal.butter(poles, [cfreq * f + delta, (cfreq + 1) * f - delta], 'bandstop', fs=Fs,
                                  analog=False, output='sos')
        mb_sos.append(sos)

    sos = scipy.signal.butter(poles, up_limit * f + delta, btype='low', fs=Fs, analog=False, output='sos')
    mb_sos.append(sos)

    mb_sos = np.vstack(mb_sos)

    return bp_sos, mb_sos


def read_data(f=10000.2, num_points=1400000):
    # Read data from CSV file
    file_path = r'E:\1 Hoc tap\Python_code\Gas\4 datagas\Average signal 3300.csv'
    Average_signal_3300 = pd.read_csv(file_path, header=None, names=['Value'])

    file_path = r'E:\1 Hoc tap\Python_code\Gas\4 datagas\Background.csv'
    Background = pd.read_csv(file_path, header=None, names=['Value'])

    file_path = r'E:\1 Hoc tap\Python_code\Gas\4 datagas\Big signal.csv'
    Big_signal = pd.read_csv(file_path, header=None, names=['Value'])

    file_path = r'E:\1 Hoc tap\Python_code\Gas\4 datagas\Small signal 1000 _2 nW.csv'
    Small_signal_1000_2nW = pd.read_csv(file_path, header=None, names=['Value'])

    file_path = r'E:\1 Hoc tap\Python_code\Gas\4 datagas\Small signal 1000.csv'
    Small_signal_1000 = pd.read_csv(file_path, header=None, names=['Value'])

    file_path = r'E:\1 Hoc tap\Python_code\Gas\4 datagas\1000ppm_1.csv'
    s1000ppm_1 = pd.read_csv(file_path, header=None, names=['Value'])
    file_path = r'E:\1 Hoc tap\Python_code\Gas\4 datagas\1000ppm_2.csv'
    s1000ppm_2 = pd.read_csv(file_path, header=None, names=['Value'])
    file_path = r'E:\1 Hoc tap\Python_code\Gas\4 datagas\1000ppm_3.csv'
    s1000ppm_3 = pd.read_csv(file_path, header=None, names=['Value'])
    file_path = r'E:\1 Hoc tap\Python_code\Gas\4 datagas\1000ppm_4.csv'
    s1000ppm_4 = pd.read_csv(file_path, header=None, names=['Value'])
    file_path = r'E:\1 Hoc tap\Python_code\Gas\4 datagas\1000ppm_5.csv'
    s1000ppm_5 = pd.read_csv(file_path, header=None, names=['Value'])

    s1_6nW = pd.read_csv(r'E:\1 Hoc tap\Python_code\Gas\4 datagas\1_6nW.csv', header=None, names=['Value'])
    s1_6nW_2 = pd.read_csv(r'E:\1 Hoc tap\Python_code\Gas\4 datagas\1_6nW_2.csv', header=None, names=['Value'])
    s1nW = pd.read_csv(r'E:\1 Hoc tap\Python_code\Gas\4 datagas\1nW.csv', header=None, names=['Value'])
    s3nW = pd.read_csv(r'E:\1 Hoc tap\Python_code\Gas\4 datagas\3nW.csv', header=None, names=['Value'])

    # Displays basic information about the data
    # print("length =", len(Average_signal_3300))
    # print("length =", len(Background))
    # print("length =", len(Big_signal))
    # print("length =", len(Small_signal_1000_2nW))
    # print("length =", len(Small_signal_1000))

    df_list = [Big_signal["Value"].to_numpy(), Average_signal_3300["Value"].to_numpy(),
               Small_signal_1000["Value"].to_numpy(), s1000ppm_1["Value"].to_numpy(), s1000ppm_2["Value"].to_numpy(),
               s1000ppm_3["Value"].to_numpy(), s1000ppm_4["Value"].to_numpy(), s1000ppm_5["Value"].to_numpy(),
               Small_signal_1000_2nW["Value"].to_numpy(), s1nW["Value"].to_numpy(), s1_6nW["Value"].to_numpy(),
               s1_6nW_2["Value"].to_numpy(), s3nW["Value"].to_numpy(),
               Background["Value"].to_numpy()]

    name_list = ["Big signal", "Average signal 3300", "Small signal 1000", "s1000ppm_1", "s1000ppm_2",
                 "s1000ppm_3", "s1000ppm_4", "s1000ppm_5",
                 "Small signal 1000 2 nW", "s1nW", "s1_6nW", "s1_6nW_2", "s3nW",
                 "Background"]

    time = (1 / 6.3e6) * np.arange(0, num_points)
    np.random.seed(42)
    gaussian_noise = np.random.normal(0, 1, num_points)
    background_noise = np.random.uniform(-0.5, 0.5, num_points)
    wave_noise = 10 * np.sin(2 * np.pi * f * time)  # Sóng có tần số f Hz

    gaussian_noise = gaussian_noise + wave_noise
    background_noise = background_noise + wave_noise

    df_list_ = df_list
    name_list_ = name_list

    df_list_.append(gaussian_noise)
    df_list_.append(background_noise)
    name_list_.append("gaussian_noise")
    name_list_.append("background_noise")

    return df_list_, name_list_


def RMSE_predict(test, predict):
    actual_values = np.array(test)
    predicted_values = np.array(predict)
    MSE = mean_squared_error(actual_values, predicted_values)
    RMSE = np.sqrt(MSE)
    return RMSE


def computer_spectrum(data_buffer, Fs: float = 3.2e5):
    N = len(data_buffer)
    nyquist_N = N // 2
    # DFT
    DFT_data = scipy.fftpack.fft(data_buffer)[:nyquist_N]
    abs_spectrum = np.abs(DFT_data)
    # Frequency
    frequencies = (Fs / N) * np.arange(nyquist_N)
    return frequencies, abs_spectrum


def computer_spectrum_sim(data_buffer):
    N = len(data_buffer)
    # DFT
    DFT_data = scipy.fft.fftshift(scipy.fft.fft(data_buffer))
    abs_spectrum = np.abs(DFT_data)
    # Frequency
    frequencies = scipy.fft.fftshift(scipy.fft.fftfreq(N))
    return frequencies, abs_spectrum


def Amp_harmonic(data: np.ndarray, up_limit: int = 15, f=10000, delta=50, fs: float = 3.2e5):
    harmonic = []
    frequencies, abs_spectrum = computer_spectrum(data, fs)
    amp_funda = np.max(abs_spectrum[index_nearest(frequencies, f - delta):index_nearest(frequencies, f + delta)])

    freq_indices = np.arange(2, up_limit, 1) * f

    for cfreq in freq_indices:
        first_index = index_nearest(frequencies, cfreq - delta)
        second_index = index_nearest(frequencies, cfreq + delta)
        harmonic.append(np.max(abs_spectrum[first_index:second_index]))

    odd_harmonic = harmonic[1::2]
    even_harmonic = harmonic[::2]
    return odd_harmonic, even_harmonic, amp_funda


def computer_THD(data: np.ndarray, up_limit: int = 15, f0=10000, fs: float = 3.2e5):
    odd_harmonic, even_harmonic, amp_funda = Amp_harmonic(data, up_limit, f0, fs)
    harmonic = np.concatenate((odd_harmonic, even_harmonic))
    harmonic_distortion = np.sqrt(np.sum(np.square(harmonic))) / amp_funda
    # THD_odd = np.sqrt(np.sum(np.square(odd_harmonic))) / amp_funda
    # THD_even = np.sqrt(np.sum(np.square(even_harmonic))) / amp_funda
    harmonic_distortion = harmonic_distortion * 100
    return np.round(harmonic_distortion, 2)


def computer_ROE(data: np.ndarray, up_limit: int = 15, f0=10000, fs: float = 3.2e5):
    odd_harmonic, even_harmonic, amp_funda = Amp_harmonic(data, up_limit, f0, fs)
    eventoodd = (np.sum(np.square(even_harmonic))) / (np.sum(np.square(odd_harmonic)))
    return np.round(eventoodd, 2)


def get_HNR(signal, rate=3.2e5, time_step=0, min_pitch=75, silence_threshold=.1, periods_per_window=3):
    # checking to make sure values are valid
    if min_pitch <= 0:
        raise ValueError("min_pitch has to be greater than zero.")
    if silence_threshold < 0 or silence_threshold > 1:
        raise ValueError("silence_threshold isn't in [ 0, 1 ].")
    # degree of overlap is four
    if time_step <= 0: time_step = (periods_per_window / 4.0) / min_pitch

    Nyquist_Frequency = rate / 2.0
    max_pitch = Nyquist_Frequency
    global_peak = max(abs(signal - signal.mean()))

    window_len = periods_per_window / float(min_pitch)

    # finding number of samples per frame and time_step
    frame_len = int(window_len * rate)
    t_len = int(time_step * rate)

    # segmenting signal, there has to be at least one frame
    num_frames = max(1, int(len(signal) / t_len + .5))

    seg_signal = [signal[int(i * t_len): int(i * t_len) + frame_len]
                  for i in range(num_frames + 1)]

    # initializing list of candidates for HNR
    best_cands = []
    for index in range(len(seg_signal)):

        segment = seg_signal[index]
        # ignoring any potential empty segment
        if len(segment) > 0:
            window_len = len(segment) / float(rate)

            # calculating autocorrelation, based off steps 3.2-3.10
            segment = segment - segment.mean()
            local_peak = max(abs(segment))
            if local_peak == 0:
                best_cands.append(.5)
            else:
                intensity = local_peak / global_peak
                window = np.hanning(len(segment))
                segment *= window

                N = len(segment)
                nsampFFT = 2 ** int(np.log2(N) + 1)
                window = np.hstack((window, np.zeros(nsampFFT - N)))
                segment = np.hstack((segment, np.zeros(nsampFFT - N)))
                x_fft = np.fft.fft(segment)
                r_a = np.real(np.fft.fft(x_fft * np.conjugate(x_fft)))
                r_a = r_a[: N]
                r_a = np.nan_to_num(r_a)

                x_fft = np.fft.fft(window)
                r_w = np.real(np.fft.fft(x_fft * np.conjugate(x_fft)))
                r_w = r_w[: N]
                r_w = np.nan_to_num(r_w)
                r_x = r_a / r_w

                r_x /= r_x[0]
                # creating an array of the points in time corresponding to the
                # sampled autocorrelation of the signal ( r_x )
                time_array = np.linspace(0, window_len, len(r_x))
                i = pu.indexes(r_x)
                max_values, max_places = r_x[i], time_array[i]
                max_place_poss = 1.0 / min_pitch
                min_place_poss = 1.0 / max_pitch

                max_values = max_values[max_places >= min_place_poss]
                max_places = max_places[max_places >= min_place_poss]

                max_values = max_values[max_places <= max_place_poss]
                max_places = max_places[max_places <= max_place_poss]

                for i in range(len(max_values)):
                    # reflecting values > 1 through 1.
                    if max_values[i] > 1.0:
                        max_values[i] = 1.0 / max_values[i]

                # eq. 23 and 24 with octave_cost, and voicing_threshold set to zero
                if len(max_values) > 0:
                    strengths = [max(max_values), max(0, 2 - (intensity /
                                                              (silence_threshold)))]
                    # if the maximum strength is the unvoiced candidate, then .5
                    # corresponds to HNR of 0
                    if np.argmax(strengths):
                        best_cands.append(0.5)
                    else:
                        best_cands.append(strengths[0])
                else:
                    best_cands.append(0.5)

    best_cands = np.array(best_cands)
    best_cands = best_cands[best_cands > 0.5]
    if len(best_cands) == 0:
        return 0
    # eq. 4
    best_cands = 10.0 * np.log10(best_cands / (1.0 - best_cands))
    best_candidate = np.mean(best_cands)
    return np.round(best_candidate, 4)


def bandpass_sos(data, bpsos):
    filtered_data = scipy.signal.sosfiltfilt(bpsos, data)
    return filtered_data


def mul_bandpass_sos(data: np.ndarray, mb_sos):
    filtered_data = scipy.signal.sosfiltfilt(mb_sos, data)
    return filtered_data


def nor_zscore(data):
    std = np.std(data)

    if std != 0:
        normalized_data = (data - np.mean(data)) / std
    else:
        normalized_data = data
    return normalized_data


def nor_minmax(data):
    min_values = np.min(data)
    max_values = np.max(data)
    normalized_data = (data - min_values) / (max_values - min_values)
    return normalized_data


def peaks_detection(data, type_peak="peak"):
    from scipy.signal import find_peaks
    y = data
    N = len(y)
    thresh_top = np.median(y) + 1 * np.std(y)  # Set thresholds

    # Find peaks
    if type_peak == "both":
        peaks, _ = find_peaks(y, height=thresh_top, prominence=0.1)
        valleys, _ = find_peaks(-y, height=thresh_top, prominence=0.1)
        peak_idx = np.sort(np.concatenate([peaks, valleys]))
    else:
        peak_idx, _ = find_peaks(y, height=thresh_top, prominence=0.1)

    amplitude = np.zeros_like(y)
    for i in range(len(peak_idx) - 1):  # lấy P-P amplitude lưu vào list amplitude
        amp_buffer = abs(y[peak_idx[i]] - y[peak_idx[i + 1]])
        amplitude[peak_idx[i]:peak_idx[i + 1]] = amp_buffer

    amplitude[:peak_idx[0]] = amplitude[peak_idx[0]]  # lấy P-P amplitude, khúc đầu vào cuối lấy cái lân cận
    amplitude[peak_idx[-1]:] = amplitude[peak_idx[-1] - 1]

    return peak_idx, amplitude


def peakfit(y, peak_idx, window_size=120):
    peak_data_list = []
    window_size = window_size // 2
    for idx in peak_idx:
        start_idx = max(0, idx - window_size)
        end_idx = min(len(y), idx + window_size + 1)

        peak_data_i = y[start_idx:end_idx]
        if len(peak_data_i) == 2 * window_size + 1:
            peak_data_list.append(peak_data_i)

    average_peak_data = np.mean(np.vstack(peak_data_list), axis=0)

    return average_peak_data


def estimation_parameter(average_peak_data):
    amplitude_ab = np.max(average_peak_data)
    x = np.arange(len(average_peak_data))
    area_under_curve = np.round(np.trapz(average_peak_data, x), 4)
    # ======================================================================
    t = np.arange(len(average_peak_data))
    centroid_moment = np.sum((t * average_peak_data) / np.sum(average_peak_data))
    centroid_amplitude = np.sum(t * average_peak_data) / np.sum(t)
    std_centroidtime = np.sqrt(np.sum(np.multiply(np.square(t - centroid_moment), average_peak_data))
                               / np.sum(average_peak_data))
    std_centroidamplitude = np.sqrt(np.sum(np.square(average_peak_data - centroid_amplitude) * t) / np.sum(t))

    time_amplitude_centroid_area = std_centroidtime * std_centroidamplitude
    # ======================================================================
    frequencies, abs_spectrum = computer_spectrum_sim(average_peak_data)
    buffer = abs_spectrum
    f = np.abs(frequencies)
    centroid_frequency = np.sum(f * buffer) / np.sum(buffer)
    centroid_amplitude = np.sum(f * buffer) / np.sum(f)

    std_centroidtime_f = np.sqrt(np.sum(np.square(f - centroid_frequency) * buffer) / np.sum(buffer))
    std_centroidamplitude_f = np.sqrt(np.sum(np.square(buffer - centroid_amplitude) * f) / np.sum(f))

    frequency_amplitude_centroid_area = std_centroidtime_f * std_centroidamplitude_f
    # ======================================================================

    return amplitude_ab, area_under_curve, time_amplitude_centroid_area, frequency_amplitude_centroid_area


def index_nearest(arr, target_value):  # tìm vị trí gần target_value
    idx = np.searchsorted(arr, target_value)
    return idx


def peaks_profile(absorption_profile, baseline, peak_idx_baseline):
    max_index = []

    # Duyệt qua mỗi phần tử trong array B
    for i in range(len(peak_idx_baseline) - 1):
        start_index = peak_idx_baseline[i]
        end_index = peak_idx_baseline[i + 1]

        # Tìm index lớn nhất trong khoảng đó và lưu vào mảng tương ứng
        buffer = np.argmax(absorption_profile[start_index:end_index])
        max_index.append(start_index + buffer)

    return np.array(max_index)


def TSMA(sigs, window_size=3, f=10000, fs=3.2e5):
    tsma = []
    num_samp_period = int(fs // f)
    all_num_period = int(len(sigs) // num_samp_period)

    for i in range(0, all_num_period - window_size + 1, 1):
        start_idx = i * num_samp_period
        end_idx = (i + window_size) * num_samp_period

        buffer = sigs[start_idx:end_idx]
        output = buffer[:num_samp_period]

        for j in range(1, window_size):
            output = np.vstack([output, buffer[j * num_samp_period:(j + 1) * num_samp_period]])

        output = np.mean(output, axis=0)
        tsma = np.concatenate([tsma, output])

    return tsma


def ATSMA(sigs):
    f = 10000.2
    fs = 6.3e6
    tsma = []
    num_samp_period = int(fs / f)
    all_num_period = int(len(sigs) / num_samp_period)
    window_size = 5
    buffer = nor_zscore(sigs)
    zcr_min = np.mean(librosa.feature.zero_crossing_rate(buffer, frame_length=4096, hop_length=1024))

    for window_size in range(5, 200, 5):

        tsma = TSMA(sigs, window_size)
        buffer = nor_zscore(tsma)

        zcr = np.mean(librosa.feature.zero_crossing_rate(buffer, frame_length=4096, hop_length=1024))

        if zcr < zcr_min:
            zcr_min = zcr
        else:
            # print("window_size=", window_size)
            return tsma
    return tsma
