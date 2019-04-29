import cv2
import os
import numpy as np
'''
url = input("URL : ")
_url = str(url)+'/shot.jpg'
def StreamVideo(_url):
    frameRequest = urllib.request.urlopen(_url) # get data from http
    framearr = np.array(bytearray(frameRequest.read()), dtype=np.uint8) # convert data to arrays 1D
    frame = cv2.imdecode(framearr, -1) # convert data to arrays 2D
    return frame
    pass
    '''
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
webcam = cv2.VideoCapture(0)
count = 0
name = input("My name : ")
while True:
	#ret, frame = webcam.read()
	#frame = StreamVideo(_url)
	status, frame = webcam.read()
	cv2.imshow('My frame',frame)
	gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
	for x,y,w,h in faces:
		facecolor =  frame[y:y+h,x:x+w]
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
		path = "Setup-face-data/ManhQuang/"+(name+"_"+str(count))+".png"
		cv2.imwrite(path,facecolor)
		count += 1
		print('Done ',str(count))
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
webcam.release()
cv2.destroyAllWindows()


