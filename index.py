# Beat tracking example  
import librosa   
# 1. Getting the file paths included audio music example  
#filename = librosa.example('nutcracker')  
filename = 'f:/python/identify-voice/sample-data/Hin_0003_Eng_m_0002.wav'  
   
 # 2. Loading the audio music like variable waveforms `y`  
 #    Storing the sample rate in variable `sr`  
y, srs = librosa.load(filename)  
  
# 3. Running the beat default tracker  
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=srs)  
  
print('Estimated tempo: {:.2f} beats per minute'.format(tempo))  
# 4. Convert the frames indice of beats event into timestamp  
beat_times = librosa.frames_to_time(beat_frames, sr=srs)
print("beat_times",beat_times)
print("Beat size",beat_times.size)
