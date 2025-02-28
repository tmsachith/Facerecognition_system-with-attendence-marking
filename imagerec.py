import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd

# Function to find encodings for known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Function to mark attendance in CSV file
def markAttendance(name):
    now = datetime.now()
    dateString = now.strftime('%Y-%m-%d')
    timeString = now.strftime('%H:%M:%S')
    
    # Create attendance file if it doesn't exist
    filename = f'Attendance_{dateString}.csv'
    if not os.path.isfile(filename):
        df = pd.DataFrame(columns=['Name', 'Time', 'Date'])
        df.to_csv(filename, index=False)
    
    # Read existing attendance
    df = pd.read_csv(filename)
    
    # Check if person is already marked for today
    if not df['Name'].str.contains(name).any():
        new_row = {'Name': name, 'Time': timeString, 'Date': dateString}
        df = df._append(new_row, ignore_index=True)
        df.to_csv(filename, index=False)
        return f"{name} - Marked"
    else:
        return f"{name} - Already Marked"

# Path to your images folder
path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)

# Load all images and names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])  # Remove file extension

print('Encoding Images...')
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    # Find all faces in current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    
    # Check each face against known encodings
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        
        # If match found, mark attendance
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            attendance_status = markAttendance(name)
            
            # Draw rectangle and name
            y1, x2, y2, x1 = faceLoc
            # Scale back up face locations since we scaled down the image
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, attendance_status, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # Unknown face
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, "Unknown", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Attendance System', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
