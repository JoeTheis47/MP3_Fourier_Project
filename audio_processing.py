import matplotlib.pyplot as plt
import numpy as np
import scipy

def subdivide(array,frame_size):
# breaks down the array into a matrix, where each row holds a segment of sample to use a fourier transform on
    subdivisions = len(array)//frame_size*2
    frame = np.zeros(frame_size)
    result = np.zeros((subdivisions,frame_size//4+1))
    window = np.hanning(frame_size//2)

    for i in range(0,subdivisions):
        frame = array[i*frame_size//2:(i+1)*frame_size//2]*window
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
        
def flux(frame,frame_begin,frame_end,peram = None):
    s_flux = []
    if frame_begin < 1:
        frame_begin = 1
    for i in range(frame_begin,frame_end):
        dev = frame[i] - frame[i-1]
        if peram is not None:
            if peram == "abs":
                dev = np.abs(dev)
            elif peram == "max":
                dev = np.maximum(dev,0)
        s_flux.append(np.sum(dev))
    return s_flux


## define variables and designate vector spaces______________________________
frame_length = 1024         # how many samples that each frame that youre analyzing has
frames_analyzed = 4         # how many frames around a guitar pick youll analyze
harmonics_observed = 17     # number of higher harmonics observed to remove some spectral leakage
spec_flux_init = .05        # when we start observing the spectral flux (for measuring buzziness)
spec_flux_end = .1          # when we stop observing the spectral flux

norm_freq = []
amplitudes = []
standard_deviation = []


## set up tools_________________________________________________________
# identify and extract the file
sample_rate,values = scipy.io.wavfile.read(r"C:\Users\joeth\Downloads\BgBsA2.wav")
sample_num = len(values)


frames,frame_quantity = subdivide(values,frame_length) # separates the values into frames of a set length to observe overall amplitude

# set up the spectral flux
tot_spectral_flux = flux(frames,1,frame_quantity, peram= "max")

# identify sudden amplitude changes higher than the standard deviation
guitar_strikes,properties = scipy.signal.find_peaks(tot_spectral_flux,prominence = np.std(tot_spectral_flux),distance = 25)
strikes = range(0,len(guitar_strikes))

ring_begin = int(sample_rate*spec_flux_init/frame_length)  # an appox. of the position that the ringing begins and ends
ring_end = int(sample_rate*spec_flux_end/frame_length)

window = np.hanning(frames_analyzed//2*frame_length)       # this is a multiplier which brings the ends of segments closer together, hile hardly
# affecting the actual processing part for the fourier transform


## the actual processing part of it_______________________________________________________
for i in strikes:
    segment = values [guitar_strikes[i]*frame_length//2 : (guitar_strikes[i] + frames_analyzed) * frame_length //2 ]
    # segment the frames into small groupings of frames after guitar strikes, and then analyze each successively

    spectral_flux = np.asarray(flux(frames,ring_begin + guitar_strikes[i],ring_end  + guitar_strikes[i],peram = "abs"))    # look at the flux between the beginning and end of the ringing
    standard_deviation.append(np.std(spectral_flux/max(spectral_flux)))    # find the standard deviation of the flux, to identify the typical jumps in amplitude in the amplitudes in the bin of the ringing
    
    segment = window*segment # this reduces the spectral decay from the jump between the start and end

    frequencies = np.array(np.fft.rfftfreq(len(segment),1/sample_rate)) # frequencies may change size depending on the number of points in each bin, so redefine every iteration

    struck_freq_amps = np.abs(np.fft.rfft(segment)) # the amplitudes of all the frequencies
    root_freq = int(autocorrelation(segment,sample_rate))   # this roughly determines the root frequency
    root_freq_idx = round(root_freq/sample_rate*len(segment))  # finds the index of the frequecy closest to the root frequency

    higher_harmonics = np.arange(1,harmonics_observed + 1)*root_freq    # create a list of higher harmonics
    harmonics_idx = [round(i*len(segment)/sample_rate) for i in higher_harmonics] # find the amplitude roughly around each of the higher harmonics


    amps = struck_freq_amps[harmonics_idx] 

    amplitudes.append(amps/np.max(amps)) # normalize the amplitudes
    norm_freq.append([frequencies[i]/root_freq for i in harmonics_idx]) # normalize the frequencies


## Testing area________________________________________________________

clarity = [(norm_freq[i]@amplitudes[i])/np.sum(amplitudes[i]) for i in strikes]
print(clarity)
print(standard_deviation)

plt.scatter(clarity,standard_deviation)
for i in strikes:
    plt.annotate(strikes[i]+1,(clarity[i],standard_deviation[i]))
plt.xlabel("clarity (bad)")
plt.ylabel("buziness (bad)")
plt.show()
'''

fig,ax = plt.subplots(ncols=len(guitar_strikes))
for i in strikes:
    ax[i].plot(norm_freq[i],amplitudes[i],'b-')
plt.show()
'''
