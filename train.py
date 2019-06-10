import librosa
import hmmlearn.hmm as hmm
import numpy as np
import pickle

def get_mfcc(filename):
    data, fs = librosa.load(filename, sr=None)
    mfcc = librosa.feature.mfcc(data, sr=fs, n_fft=1024, hop_length=128)
    return mfcc.T


if __name__ == "__main__":
    n_sample = 25
    data_trai = [get_mfcc('data/trai/trai_{}.wav'.format(i)) for i in range(n_sample)]
    data_phai = [get_mfcc('data/phai/phai_{}.wav'.format(i)) for i in range(n_sample)]
    data_xoay = [get_mfcc('data/xoay/xoay_{}.wav'.format(i)) for i in range(n_sample)]
    data_xuong = [get_mfcc('data/xuong/xuong_{}.wav'.format(i)) for i in range(n_sample)]
    data_not = [get_mfcc('data/not/not_{}.wav'.format(i)) for i in range(n_sample)]

    model_trai = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
    model_trai.fit(X=np.vstack(data_trai), lengths=[x.shape[0] for x in data_trai])
    with open("model_trai.pkl", "wb") as file: pickle.dump(model_trai, file)

    model_phai = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
    model_phai.fit(X=np.vstack(data_phai), lengths=[x.shape[0] for x in data_phai])
    with open("model_phai.pkl", "wb") as file2: pickle.dump(model_phai, file2)

    model_xoay = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
    model_xoay.fit(X=np.vstack(data_xoay), lengths=[x.shape[0] for x in data_xoay])
    with open("model_xoay.pkl", "wb") as file3: pickle.dump(model_xoay, file3)

    model_xuong = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
    model_xuong.fit(X=np.vstack(data_xuong), lengths=[x.shape[0] for x in data_xuong])
    with open("model_xuong.pkl", "wb") as file3: pickle.dump(model_xuong, file3)

    model_not = hmm.GaussianHMM(n_components=30, verbose=True, n_iter=200)
    model_not.fit(X=np.vstack(data_not), lengths=[x.shape[0] for x in data_not])
    with open("model_not.pkl", "wb") as file3: pickle.dump(model_not, file3)
