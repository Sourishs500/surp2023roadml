from firebase import firebase
from google.cloud import storage
import mmap

client = storage.Client(None, None)

file = open("fileNames.txt", 'r+')

s = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)


try:
    bucket = client.get_bucket('surp2023-5e0fb.appspot.com')
    print("success")
except:
    print("bucket not found")
 
# Iterator for all blobs in my bucket
allBlobs = list(client.list_blobs(bucket))

curClasses = ['smooth', 'fewcracks', 'manycracks']

# Iterating through all files (audio and visual)
for blob in allBlobs:
    name = blob.name
    shorterName = name[(name.index('/')+1):]
    if(len(shorterName) > 0):
        name = name.replace(':', '')

        # Adding extension for class type
        if(name.find('audio') != -1):
            for str in curClasses:
                if(name.find(str) != -1):
                    mode = name[:name.find('/')+1]
                    title = name[name.find('/'):]
                    name = mode + 'trainQuality/' + str + title

        if s.find(name.encode('utf-8')) == -1:
            with open(name, 'wb') as file_obj:
                # Downloading file to local drive
                file.write(name + '\n')
                client.download_blob_to_file(blob, file_obj)

# Getting blob using URI instead of blob object
#with open('audio/testRecording.mp3', 'wb') as file_obj:
#    client.download_blob_to_file('gs://surp2023-5e0fb.appspot.com/audio/Tue Jul 11 12:45:42 PDT 2023audio.mp3', file_obj)

file.close()





