import processSensors
import numpy as np
import matplotlib.pyplot as plt

myRoutes = processSensors.allRoutes

order = ['smooth', 'very worn']

cur = myRoutes['[(34.0646229, -118.4527842), (34.0646565, -118.4527989)]']
moreCracked = myRoutes['[(34.0646432, -118.4528042), (34.0646329, -118.4528004)]']

allGraphed = [cur, moreCracked]

plt.figure()

count = 1
for i in allGraphed:
    allAccels = []
    allTimes = []

    for pair in i:
        allAccels.append(float(pair[0])) #zAccel
        allTimes.append(int(pair[1])) #time

    t0 = allTimes[0]
    allTimes = [x - t0 for x in allTimes]

    plt.subplot(3, 1, count)
    plt.plot(allTimes, allAccels)
    plt.title(order[count-1])
    plt.xlabel('Time (milliseconds)')
    plt.ylabel('zAcceleration (m/s^2)')
    count += 1

plt.tight_layout()
plt.show()

