# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import firwin, fftconvolve


def _zoomfft_expanded_frequency_resolution(bw, dlen, win, fs, pb):
    return (bw + pb) / (dlen - win*fs/pb)


class Transform:
    '''
        2D ZoomFFT Transformer.
        
        :exmaple
            import matplotlib.pyplot as plt
            zfft = Tranfromer(1000, 2000, fs = 48000)
            x, y = zfft(signal)
            plt.stem(x,y)
    '''
    
    WIN_MAP = {
        'blackman': 5.5,
        'hanning': 3.1,
        'hamming': 3.3,
    }
    
    
    def __init__(self, freq_begin, freq_end, fs, window = 'blackman'):
        '''
        :param freq_begin: starting frequency point to zoom in. 
        :param freq_end: ending frequency point to zoom in.
        :param fs: The sampling frequency of the signal.
        :param window: FIR window function. 
        '''
        
        if freq_begin > fs:
            raise ValueError('freq_begin should not greater than max_freq')
        if freq_end > fs:
            raise ValueError('freq_end should not greater than max_freq')
        if freq_begin >= freq_end:
            freq_begin, freq_end = freq_end, freq_begin
        self._bw = freq_end - freq_begin
        self._freq_mid = (freq_begin + freq_end) / 2
        self._freq_begin = freq_begin
        self._freq_end = freq_end
        self._fs = int(fs)
        self._fir_cache = {}
        self._window = None
        self.use_window(window)
        
        
    def use_window(self, window):
        '''
        set window function for ZoomFFT FIR filter.

        :param window: window function name. (avaliable: blackman, hanning, hamming)
        '''
        self._win = Transform.WIN_MAP.get(window, None)
        if self._win is None:
            raise ValueError('unsupported window function: {}'.format(window))
        old_window = self._window
        if old_window != window:
            self._window = window
            self.clear_cache()
        
        
    def clear_cache(self):
        self._fir_cache = {}
        

    def _get_fir_filter_optimized(self, sample_len, use_cached = True):
        '''
        generate (or from cache) fir filter according to frame length.
        
        :param sample_len: length of sample data 
        '''
        cached = self._fir_cache.get(sample_len, None) if use_cached else None
        if cached:
            return cached
        
        fltr, pb = self._calc_fir_filter(self._window, sample_len)
        self._fir_cache[sample_len] = (fltr, pb)
        return fltr, pb
    
        
    def _calc_fir_filter(self, window, sample_len):
        window, win, fs, bw = self._window, self._win, self._fs, self._bw
        freq_begin, freq_end = self._freq_begin, self._freq_end
        
        # calculate length of filter (ncoff)
        pb = (win*fs + np.sqrt(win*fs*(bw*sample_len + win*fs))) / sample_len
        pb_max = None
        if freq_begin - pb/2 <= 0:
            pb_max = np.floor(pb/2 - freq_begin)
        if freq_end*2 + pb >= fs:
            pb_max2 = np.floor(pb/2 - freq_begin)
            pb_max =  pb_max2 if pb_max is None else max(pb_max, pb_max2)
        pb_min = win*fs/sample_len
        if pb > pb_min:
            pb_min = None
        if pb_max is not None or pb_min is not None:
            if pb_max is not None and pb_min is not None:
                if _zoomfft_expanded_frequency_resolution(bw, sample_len, win, fs, pb_max) < _zoomfft_expanded_frequency_resolution(bw, sample_len, win, fs, pb_min):
                    pb = pb_max
                else:
                    pb = pb_min
            else:
                pb = pb_max if pb_max is not None else pb_min
        ncoff = int(win * fs / pb)
        
        # generate FIR filter.
        return firwin(ncoff, bw, window = window , fs = fs), pb
    

    def __call__(self, sample, with_x = True):
        """
            apply transform.
        """
        
        if isinstance(sample, (list, tuple, np.ndarray)):
            sample = np.array(sample)
        else:
            raise ValueError('Unsupported sample type: {}'.format(type(sample)))
        
        # prepare
        sample_len = len(sample)
        fir, pb = self._get_fir_filter_optimized(sample_len)
        freq_mid, fs, bw = self._freq_mid, self._fs, self._bw
        bw_expanded = bw + pb
        zoom = int(fs // bw_expanded)
        freq_begin, freq_end = self._freq_begin, self._freq_end
        
        # shift
        sample = sample * np.exp(np.arange(0, sample_len, 1)*np.pi*freq_mid/fs*-2j)
        
        # filter
        sample = fftconvolve(sample, fir, mode = 'valid')
        
        # sub-sampling
        sample = fft(sample[:sample_len:zoom])*zoom
        
        # reorder
        pt_half_width = int(( bw * zoom ) / fs * len(sample) / 2)
        y = np.zeros(pt_half_width * 2, dtype = np.complex64)
        y[pt_half_width: pt_half_width*2] = sample[:pt_half_width]
        y[:pt_half_width] = sample[len(sample) - pt_half_width:]
        if not with_x:
            return y
        return np.linspace(freq_begin, freq_end, len(y)), y


def zoomfft(sample, freq_begin, freq_end, fs = None, window = 'blackman'):
    '''
    ZoomFFT
    
    :param sample: signal data.
    :param freq_begin: starting frequency point to zoom in. 
    :param freq_end: ending frequency point to zoom in.
    :param fs: The sampling frequency of the signal.
    :param window: FIR window function.
    
    :return
        spectrum_x, spectrum_y 
    '''
    
    if fs is None or fs <= 0:
        fs = len(sample)
    return Transform(freq_begin, freq_end, fs, window)(sample)