import matplotlib.pyplot as plt
import numpy as np
import librosa

def find_index_max(array):
  ## a function which returns the index of the local maximums of an input list
    output_array = np.zeros(len(array))
    for i in range(1,len(array)-1):
        try:
            if (array[i] >= array[i+1]) and (array[i] >= array[i-1]): ## if the number is greater than either of its neighbors
                output_array [i] = array[i]                           ## add it to the array
        except ValueError: ## just in case
            break
    return [i for i,val in enumerate(output_array) if val != 0] ## output all non-zero values in the array of maxes

##values,sample_rate = librosa.load("file path") ## to be included when I get the files
max_value = 10
sample_num = 100
sample_rate = int(sample_num/max_value)

## this section is just for testing purposes. Remove when actual files are being analyzed
t = np.linspace(0,10,sample_num)
values = np.sin(4*np.pi*t) + np.sin(2*np.pi*t)
fourier = np.fft.fft(values)

amplitude  = np.abs(fourier)
frequencies = np.fft.fftfreq(len(values),1/sample_rate)

index_val = find_index_max(amplitude)      ## find the index value where the amplitude of th waves are maximized
frequency_ticks =  [frequencies[index_val[num]] for num in range(0,len(index_val))] ## and put ticks on them

## Printing
ax = plt.subplot()
ax.plot(frequencies,amplitude,'b-')
ax.set_xticks(frequency_ticks) ## this is to show where the local maxes of the function are.
plt.xlabel("frequency")
plt.ylabel("Intensity")
plt.show()
