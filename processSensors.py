import firebase_admin
from firebase_admin import db
from firebase_admin import credentials
import matplotlib.pyplot as plt
import pprint
import pandas as pd
import math

file = open('sensorData.txt', 'r+')

cred = credentials.Certificate("C:\\Users\\ssour\\Downloads\\surp2023-5e0fb-firebase-adminsdk-okku0-4a0c6ee270.json")
default_app = firebase_admin.initialize_app(cred, {'databaseURL':'https://surp2023-5e0fb-default-rtdb.firebaseio.com/'})

ref = db.reference('/')

initial = ref.get()

allGPSUpdates = initial.items()

allRoutes = {}

currentAccelerations = []
curCoord = ()

count = 0


for item in allGPSUpdates:
    # item is a tuple, with item[0] being name of GPS update, item[1] being dictionary with 10 Hz update info
    currentPath = ()
    curGPSRecord = item[1]
    currentAccelerations = []
    currentPath = [(0, 0), (0, 0)]

    firstUpdate = list(curGPSRecord.keys())[0]
    lastCoord = curCoord
    
    # Find what this GPS Update's coordinate is
    curCoord = curGPSRecord[firstUpdate]['latitude'], curGPSRecord[firstUpdate]['longitude']
    
    if(count > 0):
        currentPath[1] = curCoord
        currentPath[0] = lastCoord

    
    for key in curGPSRecord:
        try:
            # Current accelerations becomes a list of tuples of z acceleration values and times that correspond to the current stretch of road (b/w GPS Updates)
            value = (curGPSRecord[key]['zAccel'], curGPSRecord[key]['otherTime'], curGPSRecord[key]['roadQuality'], curGPSRecord[key]['roadType'], curGPSRecord[key]['xAccel'], curGPSRecord[key]['yAccel'], curGPSRecord[key]['xRotation'], curGPSRecord[key]['yRotation'], curGPSRecord[key]['zRotation'])
            currentAccelerations.append(value)
        except:
            print('Fail')
    if(count > 0):
        # List is unhashable so must make currentPath (a list of tuples) into a string
        allRoutes[str(currentPath)] = currentAccelerations
    count += 1


# Dataframe, CSV file
df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in allRoutes.items() ]))
df.to_csv("coordValues.csv")

# Put in text file
niceDictionary = pprint.pformat(allRoutes)
file.write(niceDictionary)

# Close file
file.close()

