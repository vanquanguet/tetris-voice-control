import librosa
import pickle
import numpy as np
import sounddevice as sd
import soundfile as sf
import hmmlearn.hmm as hmm

def get_mfcc(filename):
    data, fs = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(data, sr=fs, n_fft=1024, hop_length=128)
    return mfcc.T

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# def record_sound(filename, duration = 1, fs=44100, play=False):
   
#     data = sd.rec(frames=duration*fs, samplerate=fs, channels=1, blocking=True)
#     if play:
#         sd.play(data, samplerate=fs, blocking=True)
#     sf.write(filename, data=data, samplerate=fs)

def record_sound(filename, duration=0.8, fs=44100, play=False):
    # sd.play( np.sin( 2*np.pi*940*np.arange(fs)/fs )  , samplerate=fs, blocking=True)
    # sd.play( np.zeros( int(fs*0.2) ), samplerate=fs, blocking=True)
    data = sd.rec(frames=int(duration*fs), samplerate=fs, channels=1, blocking=True)
    if play:
        sd.play(data, samplerate=fs, blocking=True)
    sf.write(filename, data=data, samplerate=fs)

dic_index = {0:'t',1:'p', 2:'x', 3: 'o', 4 : ''}
dic_index2 = {0:'left',1:'right', 2:'rotate', 3: 'drop', 4 : 'none'}

if __name__ == '__main__':    
    model_trai = pickle.load(open("model_trai.pkl", "rb"))
    model_phai = pickle.load(open("model_phai.pkl", "rb"))
    model_xoay = pickle.load(open("model_xoay.pkl", "rb"))
    model_xuong = pickle.load(open("model_xuong.pkl", "rb"))
    model_not = pickle.load(open("model_not.pkl", "rb"))

    # n = 25
    # for i in range(n):
    #     record_sound('data/not/not_{}.wav'.format(i))
    
    while True:
        record_sound('test.wav')
        mfcc = get_mfcc('test.wav')
        vote = [0,0,0]
        scores = [model_trai.score(mfcc), model_phai.score(mfcc), model_xoay.score(mfcc), model_xuong.score(mfcc), model_not.score(mfcc)]
        prop = softmax(scores)
        
        p_trai, p_phai, p_xoay, p_xuong, p_not = prop[0], prop[1], prop[2], prop[3], prop[4]
        # print(prop[0] + prop[1] + prop[2] + prop[3] + prop[4])
        # print(prop)
        t = 4
        if (p_trai > 0.8): 
            t = 0
        if (p_phai > 0.8): 
            t = 1
        if (p_xoay > 0.8): 
            t = 2
        if (p_xuong > 0.8):
            t = 3
        
        f = open('command.txt','w')
        f.write(dic_index[t])
        f.write(dic_index[t])
        f.write(dic_index[t])
        # print(dic_index2[t])
        f.close()

    
