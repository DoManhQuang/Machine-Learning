from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cvlib as cv
import numpy as np
import cv2
import pickle
import urllib.request
  

# read gender
model = load_model("trainner-gender/gender_detection.model")
classes = ['man','woman']
print("Read gender : OK ")

#Url Stream 
#url = input("URL : ")

def Gender_detection(frame):
    # reading face detection
    face, confidence = cv.detect_face(frame)
    # loop through detected faces
    for idx, f in enumerate(face):
        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])
        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue
        # preprocessing for gender detection model
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        # apply gender detection on face
        conf = model.predict(face_crop)[0]
        # get label with max accuracy
        idx = np.argmax(conf)
        label = classes[idx]
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)
        Y = startY - 10 if startY - 10 > 10 else startY + 10
        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)
    pass

# read face
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner-face/trainner_frontalface_alt2.yml")
print("Read face : OK ")

# gets name obj
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
    pass

#Stream camera ip
def StreamVideo(_url):
    frameRequest = urllib.request.urlopen(_url) # get data from https
    framearr = np.array(bytearray(frameRequest.read()), dtype=np.uint8) # convert data to arrays 1D
    frame = cv2.imdecode(framearr, -1) # convert data to arrays 2D
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
    #read frame from Streamcameraip
    #frame = StreamVideo(url)

    #read frame from webcam
    frame = WebcamVideo()

    #processing Gender
    Gender_detection(frame)

    #processing Face
    Face_Recognizer(frame)
    # display output
    cv2.imshow("frame", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit(0)

# release resources
video.release()
cv2.destroyAllWindows()
