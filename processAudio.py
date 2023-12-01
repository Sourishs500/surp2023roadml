import librosa
import numpy as np
import matplotlib.pyplot as plt
from soundfile import SoundFile
import os
import filetype
import noisereduce as nr
import scipy


def getFileQuantities():
    audioPath = 'C:\\Users\\ssour\\Documents\\surp2023sourishsaswade\\audio'
    photoPath = 'C:\\Users\\ssour\\Documents\\surp2023sourishsaswade\\photo'

    audioFileCount = 0
    photoFileCount = 0
    # Gathering amount of photos and audio recordings we have locally currently
    for path in os.listdir(audioPath):
        if os.path.isfile(os.path.join(audioPath, path)):
            audioFileCount += 1

    for path in os.listdir(photoPath):
        if os.path.isfile(os.path.join(photoPath, path)):
            photoFileCount += 1

    return audioFileCount, photoFileCount

def fft_plot(audio, sampling_rate):
    n = len(audio)
    T = 1/sampling_rate
    yf = scipy.fft.fft(audio)
    xf = np.linspace(0.0, 1.0/(2.0 * T), int(n/2))
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/n * np.abs(yf[:n//2]))
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    return plt.show()


audioFileCount, photoFileCount = getFileQuantities()

count = 0

filename = "C:\\Users\\ssour\\Documents\\surp2023sourishsaswade\\audio\\New Recording 31.m4a"
cutoff = 6 + filename.find('audio')
#photoFilename = "C:\\Users\\ssour\\Documents\\surp2023sourishsaswade\\photo\\" + filename[cutoff:filename.find('audio', cutoff)] + "photo.jpg"

#image = plt.imread(photoFilename)

# nrows, ncols makes the display into an array of plots
fig, ax = plt.subplots(nrows = 2, sharex=True)
y, sr = librosa.load(filename)
y2 = nr.reduce_noise(y, sr)

#fft_plot(y2, sr)

audioLength = librosa.get_duration(y=y, sr=sr)

# if an array of plots, should be ax[0] instead of ax
librosa.display.waveshow(y, sr=sr, ax=ax[0])
librosa.display.waveshow(y2, sr=sr, ax=ax[1])

ax[0].set(title = 'Audio from ' + filename)
ax[0].set_xlim([0, audioLength])
ax[0].label_outer()

ax[1].set(title = 'Denoised audio')
ax[1].label_outer()
ax[1].set_xlim([0, audioLength])

#ax[2].imshow(image)
#ax[2].set_xlim([0, 2400])
#ax[2].axis('off')

count += 1
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

"""

for path in os.listdir(audioPath):
    combinedPath = os.path.join(audioPath, path)
    if(count > 5):
        break
    if os.path.isfile(combinedPath):
        filename = combinedPath

        print(filename)
        print(filetype.guess(filename).mime)

        

        y, sr = librosa.load(filename)

        # if an array of plots, should be ax[0] instead of ax
        librosa.display.waveshow(y, sr=sr, ax=ax[count])
        ax[count].set(title='Audio from' + filename)
        ax[count].label_outer()

        count += 1
        plt.show()
"""

# check API targeted levels in android studio
# find a way to get the audio files to actually play
# check 3gpp file format in android studio









