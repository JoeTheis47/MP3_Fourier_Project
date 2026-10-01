import matplotlib.pyplot as plt
import numpy as np
import scipy

class freq_operator:
    def __init__(self,segment,sample_rate,harmonics_observed,frame_length):

        self.frequencies = np.array(np.fft.rfftfreq(len(segment),1/sample_rate)) # frequencies may change size depending on the number of points in each bin, so redefine every iteration
        self.segment = segment

        self.sample_rate = sample_rate
        self.frame_length = frame_length

        self.harmonics_observed = harmonics_observed

        self.struck_freq_amps = np.abs(np.fft.rfft(segment))
        
        self.root_frequency = float(self.root_frequency_detection())
        self.harmonics_idx = self.higher_harmonics_list()

    def higher_harmonics_list(self):
        higher_harmonics = np.arange(1,self.harmonics_observed + 1)*self.root_frequency    # create a list of higher harmonics
        return [round(i*len(self.segment)/self.sample_rate) for i in higher_harmonics if round(i*len(self.segment)/self.sample_rate) < self.frame_length] # find the amplitude roughly around each of the higher harmonics
    
    def root_frequency_detection(self):
        # takes a segment as an input, with
    
        deep_model = model()
        deep_model.load_state_dict(torch.load("guitar_frequency_model.pth"))
        deep_model.eval()

        analyzed_frames = []
        analyzed_frames.append(np.log1p(self.struck_freq_amps/(np.max(self.struck_freq_amps) + 1e-8)))

        analyzed_frames = torch.tensor(np.array(analyzed_frames)).unsqueeze(1)

        with torch.no_grad():
            pred = deep_model(analyzed_frames.float())
        predicted_hz = 2**pred.squeeze() *440

        return predicted_hz
    
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
    sample_rate,values = scipy.io.wavfile.read(r"C:\Users\joeth\OneDrive\Documents\deep_learning_learning_data\BgGsA2.wav")
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
