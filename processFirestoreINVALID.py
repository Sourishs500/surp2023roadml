import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

file = open("firestoreData.txt", "r+")

cred = credentials.Certificate("C:\\Users\\ssour\\Downloads\\surp2023-5e0fb-firebase-adminsdk-okku0-4a0c6ee270.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

curCollection = db.collection('GPSSensorValues')

docList = curCollection.get()

for i in docList:
    curDict = i.to_dict()
    curTime = firestore.SERVER_TIMESTAMP
    for key in curDict:
        file.write(key + " " + str(curDict[key]) + "\n")
    file.write("\n")



