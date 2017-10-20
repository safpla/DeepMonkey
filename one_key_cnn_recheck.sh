#! /bin/bash
cd src
source /home/xuhaowen/py3env_tf_gpu/bin/activate
echo `which python`
# Don't omit the '/' in the end
selection_table_dir='../selectionTable/autoDetection/'
output_dir='../selectionTable/recheck/'
vocalization_dir='../Vocalization/M93A/'

session_prefix='voc_M93A_c_S'
sessions_to_process=(191 198 200)
parMic=1
refMic=2
boxMic=1

selectionTable='SelectionTable_'
#for session_num in $sessions_to_process
for session_num in 191 198
do
    echo deal with session: $session_num
    wav_filename=$vocalization_dir$session_prefix$session_num.wav
    table_filename=$selection_table_dir$selectionTable$session_prefix$session_num.txt
    table_filename_new=$output_dir$selectionTable$session_prefix$session_num.txt
    python3 cnn_recheck.py session_num --parMic $parMic --refMic $refMic --boxMic $boxMic --wav_filename $wav_filename --table_filename $table_filename --table_filename_new $table_filename_new
done
