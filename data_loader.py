from torch.utils.data import Dataset
import harmonic_functions as hf
from pathlib import Path
import numpy as np

class Audio_data_set(Dataset):
    def __init__(self,audio_dir,frame_len,frames_analyzed = 4,hop_percent = 50):
        self.note_frequencies = {
        "A2": 110,
        "B3": 246.94,
        "D3": 146.83,
        "E2": 82.41,
        "E4": 329.63,
        "G3": 196.00,
        "F2": 87.31,
        "Fs2": 92.50,
        "G2": 98.00,
        "Gs2": 103.83,
        "As2": 116.54,
        "B2": 123.47,
        "C3": 130.81,
        "Cs3": 138.59,
        "Ds3": 155.56,
        "E3": 164.81,
        "F3": 174.61,
        "Fs3": 185.00,
        "Gs3": 207.65,
        "A3": 220.00,
        "As3": 233.08,
        "C4": 261.63,
        "Cs4": 277.18,
        "D4": 293.66,
        "Ds4": 311.13}

        self.frequencies = []
        self.audio_data = []

        self.frame_len = frame_len
        self.window = np.hanning(frame_len)

        file = Path(audio_dir)
        files = list(file.iterdir())

        for i in files:
            freq,data = self.open_data(i,frame_len=frame_len,frames_analyzed=frames_analyzed,hop_percent=hop_percent)
            for j in range(0,len(freq)):
                self.frequencies.append(freq[j])
                self.audio_data.append(data[j])
        self.frames_analyzed = frames_analyzed      # error withing open_data : all the data should be one long string of samples
        # dont put all the samples into a series of collected objects, but put every onset into a list, so its a size [500ish,4,2048]

    def __len__(self):
        return len(self.frequencies)
    
    def __getitem__(self,index):
        label = self.frequencies[index]
        signal = self.audio_data[index]
        return signal,label

    def open_data(self,directory,frame_len,frames_analyzed,hop_percent):
        # takes a file directory, opens all files within it and determines the note based on a _(note) at the end (not comprehensive; to add)
        # then breaks apart the files into it's plucks, and returns it all as one 
        # indexed file with evenly segmented samples and with sound data attached
        operand = hf.audio_file_operator(directory,frame_len,hop_percent,frames_analyzed)
        freq_decoder = directory.stem.split("_")[-1]
        frequency = self.note_frequencies[freq_decoder]
        onst = operand.onsets
        frame_freq = [frequency for j in range(len(onst))] # for testing purposes, this is frequency. Set to estimate root frequency once testing is concluded

        self.sample_rate = operand.sr

        return frame_freq,operand.onset_frames
