import matplotlib.pyplot as plt
import wave
from scipy.signal import spectrogram
import numpy as np
from src.config import *
class Call():
    """
    properties:
        subject
        session_name
        channel_num
        begin_time
        end_time
        call_type
        sig: optional
    """

    def __init__(self, subject, session_num, channel_num, begin_time, end_time, call_type, wav_filename, args):
        self.winsize_spec = args.winsize_spec
        self.overlap_spec = args.overlap_spec
        self.subject = subject
        self.session_num = session_num
        self.channel_num = channel_num
        self.begin_time = begin_time
        self.end_time = end_time
        self.call_type = call_type
        self.wav_filename = wav_filename
    
    def print_call(self):
        print('channel_num %s' % self.session_num)
        print('%s - %s' %(self.begin_time, self.end_time))
        print('call type %s' % self.call_type)

    def extract_spec(self, channel=0):
        if channel == 0:
            channel = self.channel_num
        if self.wav_filename == '':
            raise Exception("undefined wav filename")
        fin = wave.open(self.wav_filename, 'r')
        Fs = fin.getframerate()
        nchannels = fin.getnchannels()
        begin_time = max(0, self.begin_time - 0.1)
        end_time = self.end_time + 0.1
        fin.setpos(round(begin_time * Fs))
        sig = fin.readframes(round((end_time - begin_time) * Fs))
        sig = np.fromstring(sig, dtype=np.short)
        sig = np.reshape(sig, (-1, nchannels))
        sig = np.transpose(sig)

        _, _, spec = spectrogram(
                sig[:][channel-1],
                fs=Fs,
                window='hann',
                nperseg=self.winsize_spec,
                noverlap=self.overlap_spec,
                scaling='spectrum',
                mode='magnitude')
        spec = np.log(spec)
        # plt.imshow(spec)
        # plt.show()
        # print(self.begin_time)
        # print(self.end_time)
        fin.close()
        return spec


