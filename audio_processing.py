import matplotlib.pyplot as plt
import numpy as np
import librosa

values,sample_rate = librosa.load("file path")

fourier = np.fft.fft(values)

amplitude  = np.abs(fourier)
frequencies = np.fft.fftfreq(len(values),1/sample_rate)

plt.plot(frequencies,amplitude,'b-')
plt.show()