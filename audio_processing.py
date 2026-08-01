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

def autocorrelation(segment,sample_rate):
    # determines the root of a strum. Essentially, it adds a shifted version of the segment to itself, 
    # then looks at the maximum total of that sitting within range of the guitar. its a rough estimation of the 
    # root, seeing as you round the shift to integer valus so that they can be used as bounds as well, but its accurate enough
    autcor = np.correlate(segment,segment,mode = "full")[len(segment)-1:]

    if sample_rate > 1500:
        min_lag = sample_rate // 1300 # 1500 is the maximum output frequency of a guitar
    else:
        min_lag = 1
    
    max_lag = sample_rate // 82 # 80 is the minimum output frequency of a guitar
    lag_tot = np.argmax(autcor[min_lag:max_lag]) + min_lag

    return sample_rate/lag_tot

def closest_idx(vector,target):
    for i in range(1,len(vector)):
        if vector[i-1]<=target and vector[i]>=target:
            return i
        
def generalized_closest_idx(time,amplitude,target):
    for i in range(1,len(time)-1):
        if time[i]>=target and time[i-1]<=target:
            return np.argmax(amplitude[i-1:i+1]) + i - 1

# define variables and designate vector spaces
frame_length = 1024         # how many samples that each frame that youre analyzing has
frames_analyzed = 4         # how many frames around a guitar pick youll analyze
harmonics_observed = 17     # number of higher harmonics observed to remove some spectral leakage

norm_freq = []
amplitudes = []
tot_spectral_flux = []
standard_deviation = []


## set up tools
# identify and extract the file
sample_rate,values = scipy.io.wavfile.read(r"C:\Users\joeth\Downloads\BgBsB3.wav")
sample_num = len(values)

# detect sudden changes in amplitude
frames,frame_quantity = onset_detection(values,frame_length) # separates the values into frames of a set length to observe overall amplitude
spectral_energy = [np.sum(frames[i]) for i in range(0,frame_quantity)]

# set up the spectral flux
for i in range(1,frame_quantity):
    deviation = frames[i] - frames[i-1]     # look at the difference between frames
    tot_spectral_flux.append(np.sum(np.maximum(deviation,0))) # take all positive differences (we're looking for a positive change in flux)
    # and sum them together. A large change in amplitude can then easily be detected.

# identify sudden amplitude changes higher than the standard deviation
guitar_strikes,properties = scipy.signal.find_peaks(tot_spectral_flux,prominence = np.std(tot_spectral_flux),distance = 10)
strikes = range(0,len(guitar_strikes))

ring_begin = int(sample_rate*.05/frame_length)
ring_end = int(sample_rate*.1/frame_length)

## the actual processing part of it
for i in strikes:
    spectral_flux = []
    segment = values [guitar_strikes[i]*frame_length : (guitar_strikes[i] + frames_analyzed) * frame_length ]
    # segment the frames into small groupings of frames after guitar strikes, and then analyze each successively

    for j in range(ring_begin,ring_end):
        deviation = frames[j + guitar_strikes[i]] - frames[j + guitar_strikes[i] - 1]
        spectral_flux.append(np.sum(np.maximum(deviation,0)))
    standard_deviation.append(np.std(spectral_flux)/1000000)
    
    segment = np.hanning(len(segment))*segment # this reduces the spectral decay from the jump between the start and end

    frequencies = np.array(np.fft.rfftfreq(len(segment),1/sample_rate)) # frequencies may change size depending on the number of points in each bin, so redefine every iteration

    struck_freq_amps = np.abs(np.fft.rfft(segment)) # the amplitudes of all the frequencies
    root_freq = int(autocorrelation(segment,sample_rate))   # this roughly determines the root frequency
    root_freq_idx = closest_idx(frequencies,root_freq)  # finds the index of the frequecy closest to the root frequency

    higher_harmonics = np.arange(1,harmonics_observed + 1)*root_freq
    harmonics_idx = [generalized_closest_idx(frequencies,struck_freq_amps,i) for i in higher_harmonics]


    amps = [struck_freq_amps[i] for i in harmonics_idx]

    amplitudes.append(amps/np.max(amps)) # normalize the amplitudes
    norm_freq.append([frequencies[i]/root_freq for i in harmonics_idx]) # normalize the frequencies


## Testing area
##________________________________________________________

clarity = [(norm_freq[i]@amplitudes[i])/np.sum(amplitudes[i]) for i in strikes]
print(clarity)
print(standard_deviation)


plt.plot(clarity,standard_deviation,'*')
plt.xlabel("clarity (bad)")
plt.ylabel("buziness (bad)")
plt.show()

'''
fig,ax = plt.subplots(ncols=len(guitar_strikes))
for i in strikes:
    ax[i].plot(norm_freq[i],amplitudes[i],'b-')
plt.show()
'''
