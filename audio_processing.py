import matplotlib.pyplot as plt
import numpy as np
import scipy

def onset_detection(array,frame_size):

    subdivisions = len(array)//frame_size
    frame = np.zeros(frame_size)
    result = np.zeros((subdivisions,frame_size//2+1))
    window = np.hanning(frame_size)

    for i in range(0,subdivisions):
        frame = array[i*frame_size:(i+1)*frame_size]*window
        result[i,:] = np.abs(np.fft.rfft(frame))

    return result,subdivisions

# define variables and designate vector spaces
frame_length = 1024
frames_analyzed = 4

norm_freq = []
amplitudes = []


## set up various analysis tools
sample_rate,values = scipy.io.wavfile.read(r"C:\Users\joeth\Downloads\BgBsB3.wav")
sample_num = len(values)

frames,frame_quantity = onset_detection(values,frame_length) # separates the values into frames of a set length to observe overall amplitude
spectral_energy = np.zeros(frame_quantity)

# setting up the spectral flux, and identifying guitar strike times ______
for j in range(0,frame_quantity):
    for i in frames[j,:]:
        spectral_energy[j] += i

guitar_strikes,properties = scipy.signal.find_peaks(spectral_energy,prominence = np.std(spectral_energy),distance = 4)
strikes = range(0,len(guitar_strikes))


for i in strikes:
    segments = values [guitar_strikes[i]*frame_length : (guitar_strikes[i] + frames_analyzed) * frame_length ]
    frequencies = np.array(np.fft.rfftfreq(len(segments),1/sample_rate))

    struck_freq_amps = np.abs(np.fft.rfft(segments))
    max_freq = np.argmax(struck_freq_amps)
    amps = struck_freq_amps[max_freq:]

    amplitudes.append(amps/np.max(amps))
    norm_freq.append(frequencies[max_freq:] / frequencies[max_freq])

## Testing area
##________________________________________________________

fig,ax = plt.subplots(ncols=len(guitar_strikes))
for i in strikes:
    ax[i].plot(norm_freq[i],amplitudes[i],'b-')
plt.show()
