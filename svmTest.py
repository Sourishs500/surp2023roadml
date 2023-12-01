from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn import preprocessing
from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.kernel_based import RocketClassifier
from sklearn.ensemble import RandomForestClassifier
import librosa
import processSensors
import matplotlib.pyplot as plt
import numpy as np
import scipy
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz
import math

# max amount of updates I want in a time interval
maxTimeInterval = 100
accelValueData = []
accelTimeData = []
labels = []

smooth = []
smoothTime = []
fewcracks = []
fewcracksTime = []
manycracks = []
manycracksTime = []

allPathData = processSensors.allRoutes

for gpsPath in allPathData:
    values = allPathData[gpsPath]
    curValuesInterval = []
    curTimeInterval = []
    curQuality = ""

    
    for i in values:
        roadQuality = i[2]

        if(roadQuality == ""):
            continue
        # If the road quality changes in between GPS updates
        if(roadQuality != curQuality):
            
            if(len(curValuesInterval) > 0 and len(curTimeInterval) < maxTimeInterval):
                if(roadQuality == 'smooth'):
                    smooth = smooth + curValuesInterval
                    smoothTime = smoothTime + curTimeInterval
                elif(roadQuality == 'fewcracks'):
                    fewcracks = fewcracks + curValuesInterval
                    fewcracksTime = fewcracksTime + curTimeInterval
                elif(roadQuality == 'manycracks'):
                    manycracks = manycracks + curValuesInterval
                    manycracksTime = manycracksTime + curTimeInterval

                t0 = int(curTimeInterval[0])
                curTimeInterval = [int(x) - t0 for x in curTimeInterval]
                accelValueData.append(curValuesInterval)
                accelTimeData.append(curTimeInterval)
                labels.append(curQuality)

            curQuality = roadQuality
            curValuesInterval = []
            curTimeInterval = []

        zAccel = float(i[0])
        yAccel = float(i[5])
        xAccel = float(i[4])
        magAccel = ((zAccel ** 2) + (xAccel ** 2) + (yAccel ** 2)) ** 0.5
        adjustedAccel = (magAccel * math.sin((math.pi / 180.0) * abs(i[6]))) * (zAccel / abs(zAccel))
        adjustedAccel = yAccel * math.sin((math.pi / 180.0) * abs(i[6]))

        curValuesInterval.append(adjustedAccel)
        curTimeInterval.append(i[1])    

    if(len(curValuesInterval) > 0 and len(curTimeInterval) < maxTimeInterval):
        if(roadQuality == 'smooth'):
            smooth = smooth + curValuesInterval
            smoothTime = smoothTime + curTimeInterval
        elif(roadQuality == 'fewcracks'):
            fewcracks = fewcracks + curValuesInterval
            fewcracksTime = fewcracksTime + curTimeInterval
        elif(roadQuality == 'manycracks'):
            manycracks = manycracks + curValuesInterval
            manycracksTime = manycracksTime + curTimeInterval

        t0 = int(curTimeInterval[0])
        curTimeInterval = [int(x) - t0 for x in curTimeInterval]
        accelValueData.append(curValuesInterval)
        accelTimeData.append(curTimeInterval)
        labels.append(curQuality)



def fft_series(series, sr, time):
    N = abs(sr * (math.ceil(time[-1] - time[0])))
    T = 1/sr
    ffty = scipy.fft.fft(series)
    ffty = np.abs(ffty)
    fftx = scipy.fft.fftfreq(N, T)

    return fftx, ffty

def fft_plot(series, sr, time):
    n = len(series)
    T = 1/sr
    yf = scipy.fft.fft(series)
    xf = np.linspace(0.0, 1.0/(2.0 * T), int(n/2))
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[1].stem(xf, 2.0/n * np.abs(yf[:n//2]))
    ax[0].plot(time, series)
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    return plt.show()

plt.figure()
plt.subplot(3, 1, 1)
plt.title('smooth')
plt.plot(smoothTime, smooth)
plt.subplot(3, 1, 2)
plt.title('fewcracks')
plt.plot(fewcracksTime, fewcracks)
plt.subplot(3, 1, 3)
plt.title('manycracks')
plt.plot(manycracksTime, manycracks)
plt.tight_layout()
plt.show() 

allfftX = []
allfftY = []
newLabels = []

for chosen in range(len(accelValueData)):
    y = accelValueData[chosen]
    t = accelTimeData[chosen]
    l = labels[chosen]
    sr = 100

    if(len(t) > 1):
        ret = fft_series(y, sr, t)
        xf = ret[0]
        yf = ret[1]

        allfftX.append(xf)
        allfftY.append(yf)
        newLabels.append(l)

maxLen = 0
for value in allfftX:
    if(len(value) > maxLen):
        maxLen = len(value)

for i, value in enumerate(allfftX):
    neededZeros = [0] * (maxLen - len(value))
    allfftX[i] = list(value) + list(neededZeros)

for i, value in enumerate(allfftY):
    neededZeros = [0] * (maxLen - len(value))
    allfftY[i] = list(value) + list(neededZeros)

#2723 max length

""" allMFCC = []
for sample in accelValueData:
    mfcc = librosa.feature.mfcc(y=np.array(sample), sr=len(sample))
    
    mfccScaled = np.mean(mfcc.T, axis=0)
    allMFCC.append(np.array(mfcc).flatten()) """


values = np.asarray(allfftY)
times = (allfftX)
label = np.array(newLabels)

# dstack makes tensorData a 3-dimensional array which doesn't work with train_test_split function but it is what I need for 
# the data to be formatted correctly
""" tensorData = np.dstack((values, times))

nsamples, nx, ny = tensorData.shape
tensorDataReshaped = tensorData.reshape((nsamples,nx*ny)) """

X_train, X_test, y_train, y_test = train_test_split(values, label, test_size = 0.25, random_state=42)

clf = RandomForestClassifier()


#clf = svm.SVC(kernel='linear')
scaler = preprocessing.StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test) 

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Values shape: ", np.shape(values))
print("Times shape: ", np.shape(times))
print("Labels shape", np.shape(labels))
#print("Combined data: ", np.shape(tensorData))

print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

