import numpy as np
import cv2
import pickle
import urllib.request
import math
import time
import argparse

faceProto = "TraninerGender/opencv_face_detector.pbtxt"
faceModel = "TraninerGender/opencv_face_detector_uint8.pb"
genderProto = "TraninerGender/gender_deploy.prototxt"
genderModel = "TraninerGender/gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Nam', 'Nu']

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

# Load network
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceNet = cv2.dnn.readNet(faceModel, faceProto)

def GenderClassification(bboxes):
    padding = 20
    for bbox in bboxes:
        #print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
        max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        label = "{}".format(gender)
        cv2.putText(frameFace, label, (bbox[0], bbox[1]-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return frameFace
    pass


# read face
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner-face/trainner_frontalface_alt2.yml")
print("Read face : OK ")
# gets name object
labels = {"person_name": 0}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
print("Gets Object : OK ")
def Face_Recognizer(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert frame to frame gray
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w] 
        # recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id_, conf = recognizer.predict(roi_gray)
        if conf >3 and conf < 86:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_] # read name from labels[]
            cv2.putText(frame, name, (x,y), font, 1, (10,20,255), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
    return frame
    pass

#open video webcam
video = cv2.VideoCapture(0)
def WebcamVideo():
    status , frame = video.read()
    if not status:
        print('Connect webcam is false')
        exit(0)
    return frame
    pass

while True:
    #read frame from webcam
    frame = WebcamVideo()
    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue
    #processing Gender
    frame = GenderClassification(bboxes)

    #processing Face
    frame = Face_Recognizer(frame)
    # display output
    cv2.imshow("Result", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(0)

# release resources
video.release()
cv2.destroyAllWindows()
