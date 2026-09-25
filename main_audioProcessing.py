import matplotlib.pyplot as plt
import numpy as np
import scipy
def subdivide(array,frame_size):
# breaks down the array into a matrix, where each row holds a segment of sample to use a fourier transform on
    subdivisions = (len(array)-frame_size)//frame_size*2 # you have to remove the final frame, because its size is so varable, its likely not helpful
    result = np.zeros((subdivisions,frame_size//2+1))
    window = np.hanning(frame_size)
    for i in range(0,subdivisions):
        frame = array[ i*frame_size//2 : i*frame_size//2 + frame_size ] * window
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
    # determines the spectral flux coming from a matrix of samples
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
def onset_detection(values,frame_length,sample_rate):
# takes as an input a series of values, a length of frame that you want to break those values into, and a sample rate
# then returns the frequencies of the values subdivided in each frame, and the points where there is a large frequency spike
    guitar_strikes = []
    gather = []
    k = 0
    frames,frame_quantity = subdivide(values,frame_length) # separates the values into frames of a set length to observe overall amplitude
    # set up the spectral flux
    tot_spectral_flux = flux(frames,1,frame_quantity, peram= "max")
    # determines guitar strikes from the incoming flux of a guitar pluck
    # identify sudden amplitude changes higher than the standard deviation
    g_strikes,properties = scipy.signal.find_peaks(tot_spectral_flux,prominence = np.std(tot_spectral_flux),distance = 10)
    test = np.zeros((len(g_strikes),len(g_strikes)))
    test[0][0] = g_strikes[0]
    for i in range(1,len(g_strikes)):
        if (g_strikes[i]-g_strikes[i-1])*frame_length/sample_rate > .012:
            k += 1
        test[k,i] = (g_strikes[i])
    test = test.astype(int)
    for i in range(0,k+1):
        gather.append([j for j in test[i] if j != 0])
        root_idx = np.argmax([tot_spectral_flux[j] for j in gather[i]])
        guitar_strikes.append(gather[i][root_idx])
    return frames,guitar_strikes
class freq_operator:
    def __init__(self,segment,sample_rate,harmonics_observed,frame_length):
        self.frequencies = np.array(np.fft.rfftfreq(len(segment),1/sample_rate)) # frequencies may change size depending on the number of points in each bin, so redefine every iteration
        self.segment = segment
        self.sample_rate = sample_rate
        self.harmonics_observed = harmonics_observed
        self.frame_length = frame_length
        self.root_frequency,self.struck_freq_amps = self.root_frequency_detection()
        self.harmonics_idx = self.higher_harmonics_list()
    def higher_harmonics_list(self):
        higher_harmonics = np.arange(1,self.harmonics_observed + 1)*self.root_frequency    # create a list of higher harmonics
        return [round(i*len(self.segment)/self.sample_rate) for i in higher_harmonics if round(i*len(self.segment)/self.sample_rate) < self.frame_length] # find the amplitude roughly around each of the higher harmonics
    def root_frequency_detection(self):
        # takes a segment as an input, with
        struck_freq_amps = np.abs(np.fft.rfft(self.segment)) # the amplitudes of all the frequencies
        root_freq = int(autocorrelation(self.segment,self.sample_rate))   # this roughly determines the root frequency
        root_freq_rough_idx = round(root_freq/self.sample_rate*len(self.segment))  # finds the index of the frequecy closest to the root frequency
        root_freq_idx = root_freq_rough_idx - 2 + np.argmax(struck_freq_amps[root_freq_rough_idx-2:root_freq_rough_idx+3])
        return self.frequencies[root_freq_idx],struck_freq_amps
    def amplitude_detection(self):
        amps = np.array([np.max(self.struck_freq_amps[j-2:j+3]) for j in self.harmonics_idx])       # find the amplitudes of the harmonics by finding the peaks around
                # the estimated harmonics, then using those ( the radius for the search is 2 at the moment)
        return amps/np.max(amps) # normalize the amplitudes
if __name__ == "__main__":
## define variables and designate vector spaces______________________________
    frame_length = 2048         # how many samples that each frame that youre analyzing has
    frames_analyzed = 4         # how many frames around a guitar pick youll analyze
    harmonics_observed = 17     # number of higher harmonics observed to remove some spectral leakage
    spec_flux_init = .05        # when we start observing the spectral flux (for measuring buzziness)
    spec_flux_end = .1          # when we stop observing the spectral flux
    norm_freq = []
    amplitudes = []
    standard_deviation = []
## set up tools_________________________________________________________
# identify and extract the file
    sample_rate,values = scipy.io.wavfile.read(r"C:\Users\joeth\Downloads\sound_data\BgGsE2.wav")
    frames,guitar_strikes = onset_detection(values,frame_length,sample_rate)
    strikes = range(0,len(guitar_strikes))
    irregularity = np.zeros(len(guitar_strikes))
    ring_begin = int(sample_rate*spec_flux_init/frame_length)  # an appox. of the position that the ringing begins and ends
    ring_end = int(sample_rate*spec_flux_end/frame_length)
    window = np.hanning(frames_analyzed//2*frame_length)       # this is a multiplier which brings the ends of segments closer together, hile hardly
# affecting the actual processing part for the fourier transform
## the actual processing part of it_______________________________________________________
    for i in strikes:
        segment = window * values [guitar_strikes[i]*frame_length//2 : (guitar_strikes[i] + frames_analyzed) * frame_length //2 ]
    # segment the frames into small groupings of frames after guitar strikes, and then analyze each successively
        spectral_flux = np.asarray(flux(frames,ring_begin + guitar_strikes[i],ring_end  + guitar_strikes[i],peram = "abs"))    # look at the flux between the beginning and end of the ringing
        standard_deviation.append(np.std(spectral_flux/np.mean(spectral_flux)))    # find the standard deviation of the flux, to identify the typical jumps in amplitude in the amplitudes in the bin of the ringing
        freq_obj = freq_operator(segment,sample_rate,harmonics_observed,frame_length)
        root_freq = freq_obj.root_frequency
        amps = freq_obj.amplitude_detection()
        irregularity[i] = np.sum(np.diff(amps)**2)
        amplitudes.append(amps)
        norm_freq.append([freq_obj.frequencies[i]/root_freq for i in freq_obj.harmonics_idx]) # normalize the frequencies
## Testing area________________________________________________________
# note: atm, there are 3 possible axes: irregularity of the harmonics, the standard deviation of the spectral flux, and the harmonic centroid
# So far, Ive found the most accurate to be harmonic centroid and std of spec-flux, but spec-flux could be effected by vibrato or something
    clarity = [(norm_freq[i]@amplitudes[i])/np.sum(amplitudes[i]) for i in strikes]
    print(clarity)      # harmonic centroid
    print(standard_deviation)       # spectral flux (change in energy going out in each harmonic over time)
    print(irregularity)             # harmonic flux ()
    print(freq_obj.root_frequency)
    plt.scatter(clarity,irregularity)
    for i in strikes:
        plt.annotate(strikes[i]+1,(clarity[i],irregularity[i]))
    plt.xlabel("clarity (bad)")
    plt.ylabel("buziness (bad)")
    plt.show()
    '''
    fig,ax = plt.subplots(ncols=len(guitar_strikes))
    for i in strikes:
        ax[i].plot(norm_freq[i],amplitudes[i],'b-')
    plt.show()
    '''
## issues:
# using different notes does not result in similar scales, even if the quality is (apparently) the same
