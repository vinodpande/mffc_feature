# Beat tracking example  
import librosa
import matplotlib.pyplot as plt   
import numpy as np
import librosa.display
fig, ax = plt.subplots(nrows=2, sharex=True)

filename = 'f:/python/identify-voice/sample-data/Hin_0003_Eng_m_0002.wav'  
 # 2. Loading the audio music like variable waveforms `y`  
 #    Storing the sample rate in variable `sr`  
y, srs = librosa.load(filename)
 
mfccs = librosa.feature.mfcc(y=y, sr=srs, n_mfcc=40)
S = librosa.feature.melspectrogram(y=y, sr=srs, n_mels=128,
                                   fmax=8000)
img = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='Mel spectrogram')
ax[0].label_outer()
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='MFCC')
fig.savefig('sa5_mfcc_melspectrogram.png', bbox_inches='tight')
