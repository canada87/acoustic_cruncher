import numpy as np
from numpy.fft import fft, fftfreq, ifft
from scipy import signal
import scipy.fftpack

class FFT_transform:
    """
    class for the fourier analisys

    :param x: time coordinate
    :param y: intensity coordinate
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = len(y)

        self.freqs = fftfreq(self.n)
        self.fft_vals = fft(self.y)

        timestep = self.x[1] - self.x[0]
        timeend = self.x[-1]
        freqmin = 1/timeend
        freqmax = 1/timestep
        print('\n','#########################')
        print('timestep ', timestep, 's, freqmax ', freqmax, 'Hz')
        print('timeend ', timeend, 's, freqmin ', freqmin, 'Hz')
        print('#########################','\n')

    def trasformata(self):
        """
        perform the FFT of the signal

        :return: X[len(x)/2], Y[len(x)/2]
        """
        fft_y_proto = scipy.fftpack.fft(self.y)
        fft_y = np.zeros(round(self.n/2))
        for i in range(round(self.n/2)):
            fft_y[i] = fft_y_proto[i]

        fft_y = np.abs(fft_y)
        T = self.x[1] - self.x[0]
        fft_x = np.linspace(0, 1/(2*T), round(self.n/2))

        return fft_x, fft_y

    def power_spectral_density(self, metodo = 1, bin = 1024):
        """
        perform the Power spectral density of the signal

        :param metodo: if metodo = 1 the periodogram library is used, if metodo = 2 welch library is used
        :param bin: required only if metodo = 2
        :return: x[len(y)], y[len(y)]
        """
        if metodo == 1:
            fs = 1/((self.x[1] - self.x[0]))
            f, Pxx_den = signal.periodogram(self.y, fs)
            return f, Pxx_den
        else:
            fs = 1/((self.x[1] - self.x[0]))
            f, Pxx_den = signal.welch(self.y, fs, nperseg=bin)
            return f, Pxx_den

    def filtro(self,frequenza_rimossa=0):
        """
        it remove a single frequenzy from the data

        :param frequenza_rimossa: the frequenzy you want to remove
        :return: Y[len(y)]
        """
        nwaves = self.freqs*self.n

        fft_new = np.copy(self.fft_vals)
        fft_new[np.abs(nwaves)==frequenza_rimossa] = 0.0
        filt_data = np.real(ifft(fft_new))

        return filt_data

    def derivata(self, massimo_di_X):
        """
        it perform the derivate of the signal

        :param massimo_di_X: the X biggest value
        :return: Y[len(y)]
        """

        omg = 2.0 * np.pi/massimo_di_X

        nwaves = self.freqs*self.n
        nwaves_2pi = omg*nwaves

        yd_fft = 1.0j*nwaves_2pi*self.fft_vals
        yd_recon = np.real(ifft(yd_fft))

        return yd_recon
