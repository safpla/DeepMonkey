import numpy as np
import re
from src.config import *
import src.call as call_class
def extract_call_seg(table_filenames, args):
    call_segs = []
    for filename in table_filenames:
        subject, session_num = get_info_from_filename(filename)
        fin = open(filename, 'r')
        headline = fin.readline()
        headline = headline.strip().split('\t')
        call_seg = []
        for line in fin.readlines():
            line_split = line.strip().split('\t')
            if len(line_split) > 8:
                call_type = line_split[8]
            else:
                call_type = 'Other'
            call = call_class.Call(subject, session_num, int(line_split[2]), float(line_split[3]), float(line_split[4]),call_type, '', args)
            call_seg.append(call)
        call_segs.append(call_seg)
    return call_segs

def write_to_selectionTable(table_filenames, call_segs):
    for table_filename, call_seg in zip(table_filenames, call_segs):
        fout = open(table_filename, 'w')
        headline = 'Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tMax Freq (Hz)\tCall Type\r\n'
        fout.write(headline)
        for ind, call in enumerate(call_seg):
            line = '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\r\n' %(ind+1, 'Spectrogram 1', call.channel_num, call.begin_time, call.end_time, 4000.0, 18000.0, 18000.0, call.call_type)
            fout.write(line)


def get_info_from_filename(filename):
    filename = filename.split('/')[-1]
    divide = re.split('_|\.', filename.strip())
    if divide[0] != 'voc': 
        divide = divide[1:]
    prefix = divide[0]
    subject = divide[1]
    if divide[2] == 'c':
        session_num = divide[2] + '_' + divide[3]
    else:
        session_num = divide[2]
    return subject, session_num 




