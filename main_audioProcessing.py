## define variables and designate vector spaces______________________________
frame_length = 2048         # how many samples that each frame that youre analyzing has
frames_analyzed = 4         # how many frames around a guitar pick youll analyze
harmonics_observed = 17     # number of higher harmonics observed to remove some spectral leakage
spec_flux_init = .05        # when we start observing the spectral flux (for measuring buzziness)
spec_flux_end = .1          # when we stop observing the spectral flux
k = 0

norm_freq = []
amplitudes = []
standard_deviation = []
gather = []
guitar_strikes = []


## set up tools_________________________________________________________
# identify and extract the file
sample_rate,values = scipy.io.wavfile.read(r"C:\Users\joeth\Downloads\BgGsA2.wav")
sample_num = len(values)


frames,frame_quantity = subdivide(values,frame_length) # separates the values into frames of a set length to observe overall amplitude

# set up the spectral flux
tot_spectral_flux = flux(frames,1,frame_quantity, peram= "max")

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

    freq_obj = freq_detector(segment,sample_rate,harmonics_observed)
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
