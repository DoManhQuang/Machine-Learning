import cv2
import  numpy as np
import  math

#reading image
img = cv2.imread('cat.jpg')
img1 = cv2.imread('cat2.jpg')
img2 = cv2.imread('dog.jpg')
print('Reading done')
#create image
Arr_img = np.array(img)
Arr_img1 = np.array(img1)
Arr_img2 = np.array(img2)
print('Create image done')
#Histogram = np.arange(256)

# Count the frequency of unique values in numpy array
unique_elements, counts_elements = np.unique(Arr_img, return_counts=True) #unique_elements : biến có giá duy nhất , counts_elements số lượng của biến đó
unique_elements1, counts_elements1 = np.unique(Arr_img1, return_counts=True)
unique_elements2, counts_elements2 = np.unique(Arr_img2, return_counts=True)
print('Count the frequency of unique values in numpy array : DONE ')

count_unique = np.asarray(counts_elements)
count_unique1 = np.asarray(counts_elements1)
count_unique2 = np.asarray(counts_elements2)
print('create arrays Count the frequency of unique values : DONE ')

His_img = np.float32(count_unique)/256.0 # tính giá trị đặc trưng của ảnh
His_img1 = np.float32(count_unique1)/256.0
His_img2 = np.float32(count_unique2)/256.0

distance1 = round(np.sum(abs(His_img1 - His_img)),3)
distance2 = round(np.sum(abs(His_img2 - His_img)),3)
print(distance1)
print(distance2)

#cv2.putText(img,'This is cat',(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
#cv2.imshow('This is cat',img)
if distance1 < distance2:
    print('image 1 is cat')
    cv2.putText(img1,'image 1 is cat',(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imwrite('ImageisCat.jpg',img1)

else:
    print('image 2 is cat')
    cv2.putText(img1,'image 2 is cat',(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imwrite('ImageisCat.jpg',img2)





