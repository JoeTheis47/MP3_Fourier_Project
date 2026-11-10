import numpy as np
import librosa
from audio_analysis import audio_file_operator

class Spectral(audio_file_operator):
    def __init__(self,file_path,frame_length,frames_analyzed,hop_percent,fmin = int(librosa.note_to_hz("C2")),fmax = int(librosa.note_to_hz("C7"))):
        super().__init__(file_path, frame_length,hop_percent,frames_analyzed,fmin,fmax)

        self.harmonic_list = []
        self.amp = []
        self.amp_prev = []
        self.frequencies = []
        self.size = []
        self.norm_amp = []

        window = np.hanning((self.hop)*frames_analyzed)

        self.segment = [window * self.y[i*self.hop : (i + frames_analyzed) * self.hop] for i in self.onsets]
        seg_prev = [window * self.y[int(i*self.hop) :int((i + frames_analyzed) * self.hop)] for i in (self.onsets - np.ones(len(self.segment)))] # edge case of onst[0] = 0

        self.fft_seg = [np.abs(np.fft.rfft(self.segment[i])) for i in range(0,len(self.segment))]
        seg_prev_fft = [np.abs(np.fft.rfft(seg_prev[i])) for i in range(0,len(seg_prev))]
        # segment the frames into small groupings of frames after guitar strikes, and then analyze each successively

        for i in range(0,len(self.segment)):
            new_harmonics = self.higher_harmonics_list(self.segment[i],self.f0[i])
            self.harmonic_list.append(new_harmonics)
            start_idx = new_harmonics[0]

            freq_list = np.array(np.fft.rfftfreq(len(self.segment[i]),1/self.sr))
            self.frequencies.append(freq_list[start_idx:])

            amp = self.fft_seg[i][start_idx:]
            self.amp.append(amp)
            self.amp_prev.append(seg_prev_fft[i][start_idx:])
        
            self.norm_amp.append(np.array(amp)/np.sum(amp))

            self.size.append(len(amp))

    def Frame_energy(self,idx):

        return self.segment[idx] @ self.segment[idx]

    def spectral_centroid(self,idx):

        return self.frequencies[idx]@self.norm_amp[idx]
    
    def spectral_spread(self,idx):

        m1 = self.spectral_centroid(idx)
        return np.sqrt(np.sum((self.frequencies[idx] - m1)**2*self.norm_amp[idx]))
    
    def spectral_skewness(self,idx):

        m2 = self.spectral_spread(idx)
        if m2 == 0:
            return np.nan
        m1 = self.spectral_centroid(idx)
        return np.sum((self.frequencies[idx] - m1)**3*self.norm_amp[idx])/m2**3
    
    def spectral_kurtosis(self,idx):

        m2 = self.spectral_spread(idx)
        if m2 == 0:
            return np.nan
        m1 = self.spectral_centroid(idx)
        return np.sum((self.frequencies[idx] - m1)**4*self.norm_amp[idx])/m2**4
    
    def spectral_slope(self,idx):

        freq_dev = self.frequencies[idx] - np.mean(self.frequencies[idx])
        amp_dev = self.amp[idx] - np.mean(self.amp[idx])
        return amp_dev @ freq_dev/np.sum(freq_dev**2)
    
    def spectral_decrease(self,idx):

        amp_dif = np.abs(self.amp[idx][1:] - self.amp[idx][0])
        return np.sum(amp_dif)/(self.size[idx]-1)/np.sum(self.amp[idx][1:])
        
    def spectral_rolloff(self,idx,ratio = None):
        # find the frequency at which the ratio amount of energy is below
        if ratio is None:
            ratio = .95
        rolloff_idx = np.searchsorted(np.cumsum(self.fft_seg[idx]),np.cumsum(self.fft_seg[idx])[-1]*ratio) # this is a clever bit of code: it looks for the index at which
        # ratio% of the total sum of the fft is at. note that both use cumsum because the cumsum will have the same errors, meaning it will actually give a correct answer

        return self.frequencies[idx][min(rolloff_idx,len(self.frequencies[idx]) - 1)]

    def spectral_variation(self,idx):

        tot_cur = 0
        tot_prev = 0
        tot_comb = 0

        for i in range(0,self.size[idx]):
            tot_cur+=self.amp[idx][i]**2
            tot_prev += self.amp_prev[idx][i]**2
            tot_comb += self.amp[idx][i]*self.amp_prev[idx][i]

        return 1 - tot_comb/np.sqrt(tot_cur)/np.sqrt(tot_prev)

    def spectral_flatness(self,idx):
        geometric_mean = np.exp(np.mean(np.log(self.amp[idx])))
        return geometric_mean / np.mean(self.amp[idx])
    
    def spectral_crest(self,idx):
        return np.max(self.amp[idx])*float(self.size[idx])/np.sum(self.amp[idx])

    def spectral_flux(self,idx):
        return np.sqrt(np.sum(np.abs(self.amp[idx] - self.amp_prev[idx])@np.abs(self.amp[idx] - self.amp_prev[idx])))/self.size[idx]

    def contents(self,meas_type = "ALL"):
        meas = []
        out = []

        meas.append(np.array([self.spectral_flux(i) for i in range(0,len(self.segment))]))
        meas.append(np.array([self.spectral_crest(i)  for i in range(0,len(self.segment))]))
        meas.append(np.array([self.spectral_flatness(i)  for i in range(0,len(self.segment))]))
        meas.append(np.array([self.spectral_variation(i)  for i in range(0,len(self.segment))]))
        meas.append(np.array([self.spectral_rolloff(i)  for i in range(0,len(self.segment))]))
        meas.append(np.array([self.spectral_decrease(i)  for i in range(0,len(self.segment))]))
        meas.append(np.array([self.spectral_slope(i)  for i in range(0,len(self.segment))]))
        meas.append(np.array([self.spectral_skewness(i)  for i in range(0,len(self.segment))]))
        meas.append(np.array([self.spectral_kurtosis(i)  for i in range(0,len(self.segment))]))
        meas.append(np.array([self.spectral_spread(i)  for i in range(0,len(self.segment))]))
        meas.append(np.array([self.spectral_centroid(i)  for i in range(0,len(self.segment))]))
        meas.append(np.array([self.Frame_energy(i)  for i in range(0,len(self.segment))]))

        for i in range(0,12):
            if meas_type == "ALL":
                out.append(meas[i].tolist())
            elif meas_type == "Mean":
                out.append(np.mean(meas[i]))
            elif meas_type == "Median":
                out.append(np.median(meas[i]))
            elif meas_type == "Min":
                out.append(min(meas[i]))
            elif meas_type == "Max":
                out.append(max(meas[i]))
            else:
                raise ValueError("Invalid measurement type. Valid types are ALL,Mean,Median,Min,Max")

        print(f"spectral_flux = {out[0]}")
        print(f"spectral_crest = {out[1]}")
        print(f"spectral_flatness = {out[2]}")
        print(f"spectral variation = {out[3]}")
        print(f"spectral rolloff = {out[4]}")
        print(f"spectral decrease = {out[5]}")
        print(f"spectral slope = {out[6]}")
        print(f"spectral skewness = {out[7]}")
        print(f"spectral kurtosis = {out[8]}")
        print(f"spectral spread = {out[9]}")
        print(f"spectral centroid = {out[10]}")
        print(f"frame energy = {out[11]}")

    def __len__(self):
        # returns the number of strikes in the input file
        return len(self.segment)

class Harmonic(audio_file_operator):
    def __init__(self,file_path,frame_length,frames_analyzed,hop_percent,fmin = int(librosa.note_to_hz("C2")),fmax = int(librosa.note_to_hz("C7"))):
        super().__init__(file_path,frame_length,hop_percent,frames_analyzed,fmin,fmax)

        self.harmonic_list = []
        self.amp = []
        self.frequencies = []
        self.amp_prev = []
        self.size = []
        self.norm_amp = []
        self.harmonic_amp = []
        self.harm_amp_prev = []
        self.partial_energy = []
        self.total_energy = []
        self.harmonic_energy = []
        self.noisiness = []
        self.harmonic_list = []
        self.partial_energy = []
        self.total_energy = []
        self.harmonic_energy = []
        self.harmonics_freq = []

        window = np.hanning(self.hop*frames_analyzed)
        self.segment = [window * self.y[i*self.hop : (i + frames_analyzed) * self.hop] for i in self.onsets]
        self.seg_prev = [window * self.y[(i-1)*self.hop : (i + frames_analyzed -1) * self.hop] for i in self.onsets]

        self.corr_f0 = self.f0

        self.fft_seg = [np.abs(np.fft.rfft(self.segment[i])) for i in range(0,len(self.segment))]
        self.fft_seg_prev = [np.abs(np.fft.rfft(self.seg_prev[i])) for i in range(0,len(self.seg_prev))]
        # segment the frames into small groupings of frames after guitar strikes, and then analyze each successively

        for i in range(0,len(self.segment)):
            new_harmonics = self.higher_harmonics_list(self.segment[i],self.corr_f0[i])
            self.harmonic_list.append(new_harmonics)
            start_idx = new_harmonics[0]

            freq_list = np.array(np.fft.rfftfreq(len(self.segment[i]),1/self.sr))
            self.frequencies.append(freq_list[start_idx:])
            self.harmonics_freq.append(freq_list[new_harmonics])

            amp = self.fft_seg[i][start_idx:]
            self.amp.append(amp)
            self.harmonic_amp.append(self.fft_seg[i][new_harmonics])
            self.harm_amp_prev.append(self.fft_seg_prev[i][new_harmonics])
        
            self.norm_amp.append(np.array(amp)/np.sum(amp))

            self.size.append(len(amp))
            self.partial_energy.append(self.amp[i]@self.amp[i])
            self.total_energy.append(self.segment[i] @ self.segment[i])

            self.harmonic_energy.append(self.harmonic_amp[i] @ self.harmonic_amp[i])

        self.noisiness = [(self.total_energy[i] - self.partial_energy[i])/self.total_energy[i] for i in range(0,len(self.total_energy))]

    def harmonic_oddToEven(self,idx):
        even = self.harmonic_amp[idx][1::2]
        odd = self.harmonic_amp[idx][0::2]
        return sum(odd)/sum(even)
    
    def freq_norm_harmonic_centroid(self,idx):
        return self.harmonic_amp[idx] @ self.harmonics_freq[idx] / np.sum(self.harmonic_amp[idx]) / self.corr_f0[idx]
    
    def harmonic_spread(self,idx):
        mean_ratio = (self.harmonic_amp[idx]**2 @ self.harmonics_freq[idx])/self.corr_f0[idx]/np.sum(self.harmonic_amp[idx]**2)
        return np.sqrt((self.harmonic_amp[idx]**2 @ (self.harmonics_freq[idx]/self.corr_f0[idx] - mean_ratio) **2)/np.sum(self.harmonic_amp[idx]**2))
    
    def harmonic_spectral_variation(self,idx):
        current = self.harmonic_amp[idx]/np.linalg.norm(self.harmonic_amp[idx])
        prev = self.a=self.harm_amp_prev[idx]/np.linalg.norm(self.harm_amp_prev[idx])
        return 1 - np.clip(current@prev,-1,1)

    def harmonic_deviation(self,idx):
        HDEV = 0
        for i in range(1,len(self.harmonic_list[idx])-1):
            SE = (self.harmonic_amp[idx][i-1] + self.harmonic_amp[idx][i] + self.harmonic_amp[idx][i+1])/3
            HDEV += (self.harmonic_amp[idx][i] - SE)/(len(self.harmonic_list[idx])-2)
        return HDEV

    def tristimulus_measures(self,idx):
        tot = np.sum(self.harmonic_amp[idx])
        t1 = self.harmonic_amp[idx][0]/tot
        t2 = np.sum(self.harmonic_amp[idx][1:4])/tot
        t3 = np.sum(self.harmonic_amp[idx][4:])/tot
        return t1,t2,t3

    def inharmonicity(self,idx):
        dp = [self.harmonics_freq[idx][i] - (i+1)*self.corr_f0[idx] for i in range(0,len(self.harmonics_freq[idx]))] # the absolute value is not in timbre toolbox, just added to prevent opposite signs from canceling out
        return dp @ self.harmonic_amp[idx]**2/np.sum(self.harmonic_amp[idx]**2)

    def contents(self,meas_type = "ALL"):
        meas = []
        out = []
        list_1 = []
        list_2 = []
        list_3 = []

        trist = np.array([self.tristimulus_measures(i) for i in range(0,len(self.segment))])
        for meas_1,meas_2,meas_3 in trist:
            list_1.append(meas_1)
            list_2.append(meas_2)
            list_3.append(meas_3)

        meas.append(np.array([self.harmonic_oddToEven(i) for i in range(0,len(self.segment))]))
        meas.append(np.array([self.freq_norm_harmonic_centroid(i) for i in range(0,len(self.segment))]))
        meas.append(np.array([self.harmonic_spread(i) for i in range(0,len(self.segment))]))
        meas.append(np.array([self.harmonic_spectral_variation(i) for i in range(0,len(self.segment))]))
        meas.append(np.array([self.harmonic_deviation(i) for i in range(0,len(self.segment))]))
        meas.append(np.array(list_1))
        meas.append(np.array(list_2))
        meas.append(np.array(list_3))
        meas.append(np.array([self.inharmonicity(i) for i in range(0,len(self.segment))]))
        for i in range(0,9):
            if meas_type == "ALL":
                out.append(meas[i].tolist())
            elif meas_type == "Mean":
                out.append(np.mean(meas[i]))
            elif meas_type == "Median":
                out.append(np.median(meas[i]))
            elif meas_type == "Min":
                out.append(min(meas[i]))
            elif meas_type == "Max":
                 out.append(max(meas[i]))
            else:
                raise ValueError("Invalid measurement type. Valid types are ALL,Mean,Median,Min,Max")

        if meas_type == "ALL":
            trist_meas = zip(out[5],out[6],out[7])
            trist_meas = list(trist_meas)
        else:
            trist_meas = [out[5].tolist(),out[6].tolist(),out[7].tolist()]

                
        # using numpy arrays to get the tolist function, so that dtype isnt shown. To know the dtype, remove .tolist()
        print(f"harmonic odd To Even relationships: {out[0]}")
        print(f"frequency normalized harmonic centroid: {out[1]}")
        print(f"harmonic spread: {out[2]}")
        print(f"harmonic spectral variation: {out[3]}")
        print(f"harmonic deviation: {out[4]}")
        print(f"tristimulus measures: {trist_meas}")
        print(f"inharmonicity: {out[8]}")

    def __len__(self):
        # returns the number of strikes in the input file
        return len(self.segment)


if __name__ == "__main__":
    spectral = Spectral(r"C:\Users\joeth\OneDrive\Documents\deep_learning_data\deep_learning_learning_data\Gs_As2.wav",hop_percent=50,frame_length = 2048,frames_analyzed=4)
    harmonics = Harmonic(r"C:\Users\joeth\OneDrive\Documents\deep_learning_data\deep_learning_learning_data\Gs_As2.wav",hop_percent=50,frame_length = 2048,frames_analyzed= 4)
    spectral.contents("Median")
    harmonics.contents("Median")
