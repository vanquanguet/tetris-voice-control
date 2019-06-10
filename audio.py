'''
Requirements:
+ pyaudio - `pip install pyaudio`
+ py-webrtcvad - `pip install webrtcvad`
'''
import webrtcvad
import collections
import sys
import signal
import pyaudio

from array import array
from struct import pack
import wave
import time
import pickle as pkl
from math import exp
import librosa
import numpy as np

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30       # supports 10, 20 and 30 (ms)
PADDING_DURATION_MS = 1500   # 1 sec jugement
CHUNK_SIZE = 16*30 #int(RATE * CHUNK_DURATION_MS / 1000)  # chunk to read
CHUNK_BYTES = CHUNK_SIZE * 2  # 16bit = 2 bytes, PCM
NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
# NUM_WINDOW_CHUNKS = int(240 / CHUNK_DURATION_MS)
NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)  # 400 ms/ 30ms  ge
NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS+20

START_OFFSET = int(NUM_WINDOW_CHUNKS * CHUNK_DURATION_MS * 0.5 * RATE)

vad = webrtcvad.Vad(1)

pa = pyaudio.PyAudio()
stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 start=False,
                 # input_device_index=2,
                 frames_per_buffer=CHUNK_SIZE)


got_a_sentence = False
leave = False
model_trai = pkl.load(open('model/model_trai.pkl',"rb"))
model_phai = pkl.load(open('model/model_phai.pkl',"rb"))
model_xoay = pkl.load(open('model/model_xoay.pkl',"rb"))
model_xuong = pkl.load(open('model/model_xuong.pkl',"rb"))

def handle_int(sig, chunk):
    global leave, got_a_sentence
    leave = True
    got_a_sentence = True


def record_to_file(path, data, sample_width):
    "Records from the microphone and outputs the resulting data to 'path'"
    # sample_width, data = record()
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 32767  # 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r
def get_mfcc(filename):
    data, fs = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(data, sr=fs, n_fft=1024, hop_length=128)
    return mfcc.T

def get_prob(log_x1, log_x2):
    if log_x1 < log_x2:
        exp_x1_x2 = exp(log_x1-log_x2)
        return exp_x1_x2 / (1+exp_x1_x2), 1 / (1+exp_x1_x2)
    else:
        p = get_prob(log_x2, log_x1)
        return p[1], p[0]
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def process():
    dic_index = {0:'t',1:'p', 2:'x', 3: 'o', 4 : ''}
    dic_index2 = {0:'left',1:'right', 2:'rotate', 3: 'drop', 4 : 'none'}
    mfcc = get_mfcc("recording.wav")
    vote = [0,0,0]
    scores = [model_trai.score(mfcc), model_phai.score(mfcc), model_xoay.score(mfcc), model_xuong.score(mfcc)]
    prop = softmax(scores)
        
    p_trai, p_phai, p_xoay, p_xuong = prop[0], prop[1], prop[2], prop[3]
    t = np.argmax(prop)

    t = 4
    if (p_trai > 0.99): 
        t = 0
    if (p_phai > 0.99): 
        t = 1
    if (p_xoay > 0.99): 
        t = 2
    if (p_xuong > 0.99):
        t = 3
        
    f = open('command.txt','w')
    f.write(dic_index[t])
    f.write(dic_index[t])
    f.write(dic_index[t])
    
    f.close()

signal.signal(signal.SIGINT, handle_int)

data_len = 0
while not leave:
    ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
    triggered = False
    voiced_frames = []
    ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
    ring_buffer_index = 0

    ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
    ring_buffer_index_end = 0
    buffer_in = ''
    # WangS
    raw_data = array('h')
    index = 0
    start_point = 0
    StartTime = time.time()
    print("* recording: ")
    stream.start_stream()
    mystat = 0
    while not got_a_sentence and not leave:
        chunk = stream.read(CHUNK_SIZE)
        # add WangS
        raw_data.extend(array('h', chunk))
        index += CHUNK_SIZE
        TimeUse = time.time() - StartTime

        active = vad.is_speech(chunk, RATE)
        if active and mystat == 0:
            mystat = index-CHUNK_SIZE
        if not active:
            mystat = 0
        sys.stdout.write('1' if active else '_')
        ring_buffer_flags[ring_buffer_index] = 1 if active else 0
        ring_buffer_index += 1
        ring_buffer_index %= NUM_WINDOW_CHUNKS

        ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
        ring_buffer_index_end += 1
        ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END

        # start point detection
        if not triggered:
            ring_buffer.append(chunk)
            num_voiced = sum(ring_buffer_flags)
            if num_voiced > 0.9 * NUM_WINDOW_CHUNKS:
                sys.stdout.write(' Open ')
                triggered = True
                #start_point = mystat - CHUNK_SIZE  # start point
                start_point = index - CHUNK_DURATION_MS*CHUNK_SIZE
                # voiced_frames.extend(ring_buffer)
                ring_buffer.clear()
        # end point detection
        else:
            # voiced_frames.append(chunk)
            ring_buffer.append(chunk)
            num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)
            if num_unvoiced > 0.8*NUM_WINDOW_CHUNKS_END :
                sys.stdout.write(' Close ')
                triggered = False
                raw_data.reverse()
                for index in range(start_point-1):
                    if len(raw_data) <= 0:
                        break
                    raw_data.pop()
                if len(raw_data) <= 0:
                        continue
                raw_data.reverse()
                raw_data = normalize(raw_data)
                record_to_file("recording.wav", raw_data, 2)
                data_len+=1
                ring_buffer.clear()
                index = 0
                start_point = 0
                raw_data = array('h')
                process()
                ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
                triggered = False
                voiced_frames = []
                ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
                ring_buffer_index = 0

                ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
                ring_buffer_index_end = 0
                buffer_in = ''
                # WangS
                index = 0
                

        sys.stdout.flush()

    sys.stdout.write('\n')
    # data = b''.join(voiced_frames)

    stream.stop_stream()
    print("* done recording")
    got_a_sentence = False

    # write to file
    # raw_data.reverse()
    # for index in range(start_point):
    #     raw_data.pop()
    # raw_data.reverse()
    # raw_data = normalize(raw_data)
    # record_to_file("recording.wav", raw_data, 2)
    # leave = True

stream.close()