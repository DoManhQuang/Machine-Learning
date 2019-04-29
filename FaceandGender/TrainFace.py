import cv2
import os
import numpy as np
from PIL import Image
import pickle
import argparse

# handle command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--folder', help='path to folder dataset')
args = ap.parse_args()
name_folder = args.folder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, name_folder)

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            pil_image = Image.open(path).convert("L")
            image_array = np.array(pil_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
with open("pickles/face-labels.pickle", 'wb') as f: # Đánh chỉ mục dữ liệu đặc trưng
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.write("trainner-face/trainner_frontalface_alt2.yml")
print("__Finish__")