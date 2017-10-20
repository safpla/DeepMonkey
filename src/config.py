MAX_LSTM_STEP = 100 # 4 secs: 160 * 0.05 * 0.8
DATA_HEIGHT = 64
DATA_WIDTH = 64


# CALLTYPE_IND_DIC = {'Noise': 0, 'Trill': 1, 'Phee': 2, 'Trillphee': 3, 'Twitter':4, 'Sd-peep': 5, 'Others': 6}
CALLTYPE_IND_DIC = {'Noise': 0, 'Sd-peep': 1, 'Tsik': 2, 'P-peep': 3, 'Phee': 4, 'Trillphee': 5, 'Trill': 6, 'Tse': 7, 'Twitter': 8, 'Others': 9}
IND_CALLTYPE_DIC = {0: 'Noise', 1: 'Sd-peep', 2: 'Tsik', 3: 'P-peep', 4: 'Phee', 5: 'Trillphee', 6: 'Trill', 7: 'Tse', 8: 'Twitter', 9: 'Others'}

import argparse

def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('session_num', help='session number', type=str)
    parser.add_argument('--parMic', type=int, default=1, help='the index of parabolic mic channel')
    parser.add_argument('--refMic', type=int, default=2, help='the index of reference mic channel')
    parser.add_argument('--boxMic', type=int, default=1, help='the index of the channel where the bounding boxes are in')
    parser.add_argument('--winsize_spec', type=int, default=512, help='number of points used in STFT, 256 points in frequency domain')
    parser.add_argument('--overlap_spec', type=int, default=462, help='number of points shifted in STFT')
    parser.add_argument('--winsize_frame', type=float, default=0.05, help='sec, 50 points in spectrum if FS = 50000 and SHIFT_SPEC = 50')
    parser.add_argument('--shift_frame', type=float, default=0.8, help='persent')
    parser.add_argument('--FS', type=int, default=50000, help='fs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--wav_filename', type=str, help='path of wave file')
    parser.add_argument('--table_filename', type=str, help='path of input selection table')
    parser.add_argument('--table_filename_new', type=str, help='path of output selection table')
    args = parser.parse_args()
    return args

