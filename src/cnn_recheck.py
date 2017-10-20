import wave
import argparse
import random
import time
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from src.config import *
import h5py
import src.call as call
import src.utilizer as utilizer
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import matplotlib.image as pyimg
from PIL import Image
import src.predictor as predictor_class
import tensorflow as tf

def extract_data(wav_filenames, call_segs, hdf5_filename):
    for filename, call_seg in zip(wav_filenames, call_segs):
        fin = wave.open(filename, 'r')
        Fs = fin.getframerate()
        nframes = fin.getnframes()
        nchannels = fin.getnchannels()

        oneSampleFrames = round(((MAX_LSTM_STEP - 1) * SHIFT_FRAME + 1) * WINSIZE_FRAME * Fs)
        for i in range(int(nframes / oneSampleFrames)):
            if sampleCount >= 2000:
                break
            startTime = i * oneSampleFrames / Fs
            stopTime = (i + 1) * oneSampleFrames / Fs
            specLabel = audio_state_query.query(startTime, stopTime, MAX_LSTM_STEP)
            if sum(specLabel) < 3:
                survive = random.uniform(0,1)
                if survive < 0.9:
                    continue
                pure_noise_sample += 1

            sig = fin.readframes(oneSampleFrames)
            sig = np.fromstring(sig, dtype=np.short)
            sig = np.reshape(sig, (-1, nchannels))
            sig = np.transpose(sig)

            height = DATA_HEIGHT
            width = round(((MAX_LSTM_STEP -1) * SHIFT_FRAME + 1) * DATA_WIDTH)

            _, _, specPar = spectrogram(
                    sig[:][0],
                    fs=Fs, window='hann',
                    nperseg=WINSIZE_SPEC,
                    noverlap=OVERLAP_SPEC,
                    scaling='spectrum',
                    mode='magnitude')
            specPar = np.log(specPar)
            specParPad = cv2.resize(specPar, (width, height))

            _, _, specRef = spectrogram(
                    sig[:][1],
                    fs=Fs, window='hann',
                    nperseg=WINSIZE_SPEC,
                    noverlap=OVERLAP_SPEC,
                    scaling='spectrum',
                    mode='magnitude')
            specRef = np.log(specRef)
            specRefPad = cv2.resize(specRef, (width, height))

            specDifPad = specParPad - specRefPad

            for j in range(MAX_LSTM_STEP):
                buf[:,:,0,j] = specParPad[:,j*int(DATA_WIDTH * SHIFT_FRAME) : j*int(DATA_WIDTH * SHIFT_FRAME) + DATA_WIDTH]
                buf[:,:,1,j] = specRefPad[:,j*int(DATA_WIDTH * SHIFT_FRAME) : j*int(DATA_WIDTH * SHIFT_FRAME) + DATA_WIDTH]
                buf[:,:,2,j] = specDifPad[:,j*int(DATA_WIDTH * SHIFT_FRAME) : j*int(DATA_WIDTH * SHIFT_FRAME) + DATA_WIDTH]
                buf_label[j] = specLabel[j]
            print('write sample %g' % sampleCount)
            dset_data[sampleCount,:,:,:,:] = buf
            dset_label[sampleCount,:] = buf_label
            sampleCount += 1
    sizeData[0] = sampleCount
    sizeLabel[0] = sampleCount
    dset_data.resize(sizeData)
    dset_label.resize(sizeLabel)
    fout.close()
    print(pure_noise_sample)

def minmaxnorm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


if __name__ == '__main__':
    args = load_arguments()
    session_num = args.session_num
    parMic = args.parMic
    refMic = args.refMic
    boxMic = args.boxMic
    wav_filenames = [args.wav_filename]
    table_filenames = [args.table_filename]
    table_filenames_new = [args.table_filename_new]

    #wav_filenames = ['../Vocalization/M93A/voc_M93A_c_S' + session_num + '.wav']
    #table_filenames = ['../selectionTable/autoDetection/SelectionTable_voc_M93A_c_S' + session_num + '.txt']
    #table_filenames_new = ['../selectionTable/recheck/SelectionTable_voc_M93A_c_S' + session_num + '.txt']
    model_file = '../Models/M93A_9606_tc/-1100'
    call_segs = utilizer.extract_call_seg(table_filenames, args)
    configproto = tf.ConfigProto()
    configproto.gpu_options.allow_growth = True
    configproto.allow_soft_placement = True
    call_segs_new = []
    with tf.Session(config=configproto) as sess:
        predictor = predictor_class.Predictor()
        predictor.load_model(sess, model_file)
        for wav_filename, call_seg in zip(wav_filenames, call_segs):
            call_seg_new = []
            for call in call_seg:
                call.wav_filename = wav_filename
                specPar = call.extract_spec(parMic)
                specRef = call.extract_spec(refMic)
                specPar = Image.fromarray(specPar)
                specRef = Image.fromarray(specRef)
                specPar = specPar.resize((224,224))
                specRef = specRef.resize((224,224))
                specPar = np.asarray(specPar)
                specRef = np.asarray(specRef)
                specDif = specPar - specRef
                specPar = minmaxnorm(specPar)
                specRef = minmaxnorm(specRef)
                specDif = minmaxnorm(specDif)

                batch_xs = np.zeros([1,224,224,3])
                batch_xs[0,:,:,0] = specPar
                batch_xs[0,:,:,1] = specRef
                batch_xs[0,:,:,2] = specDif
                label = predictor.predict(sess, batch_xs)
                label = label[0]
                print('%s - %s     origin: %s, predict: %s' %(call.begin_time, call.end_time, call.call_type, IND_CALLTYPE_DIC[label]))

                if label > 0:
                    call.call_type = IND_CALLTYPE_DIC[label]
                    call_seg_new.append(call)
            print(type(call_seg_new))
            call_segs_new.append(call_seg_new)
    utilizer.write_to_selectionTable(table_filenames_new, call_segs_new)




