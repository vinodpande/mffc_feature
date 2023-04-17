# Beat tracking example  
import librosa   
# 1. Getting the file paths included audio music example  
#filename = librosa.example('nutcracker')  
filename = 'f:/python/identify-voice/sample-data/Audio_File_2.wav'  
 # 2. Loading the audio music like variable waveforms `y`  
 #    Storing the sample rate in variable `sr`  
y, srs = librosa.load(filename)  
time_series = librosa.feature.mfcc(y=y, sr=srs) 
print('time_series' ,time_series)  

