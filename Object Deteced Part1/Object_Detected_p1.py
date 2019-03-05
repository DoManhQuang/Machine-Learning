import cv2
import  numpy as np
import  math


'''
Trong xử lý ảnh sử dụng tính đặc trưng (histogram) của màu sắc để xác định đối tượng sẽ gây 
ra sai lệch rất nhiều đối với các màu sắc gần tương đồng nhau . VD như Mèo có màu đen sẽ giống
con chó có màu đen hơn con mèo màu vàng . Để xứ lý vấn đề đó ta có thể tăng nhiều đối tượng
ở dataset và so sánh kỳ vọng của ảnh đầu vào .
'''
#reading image
cat = cv2.imread('cat.jpg')
img1 = cv2.imread('dataset/cat1.jpg')
img2 = cv2.imread('dataset/cat2.jpg')
img3 = cv2.imread('dataset/cat3.jpg')
img4 = cv2.imread('dataset/cat4.jpg')
img5 = cv2.imread('dataset/cat5.jpg')
img6 = cv2.imread('dataset/cat6.jpg')
dog = cv2.imread('dog.jpg')
print('Reading file : Done')

#create image gray
gray_cat =  cv2.cvtColor(cat,cv2.COLOR_BGR2GRAY)
gray_dog =  cv2.cvtColor(dog,cv2.COLOR_BGR2GRAY)
gray_img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
gray_img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
gray_img4 = cv2.cvtColor(img4,cv2.COLOR_BGR2GRAY)
gray_img5 = cv2.cvtColor(img5,cv2.COLOR_BGR2GRAY)
gray_img6 = cv2.cvtColor(img6,cv2.COLOR_BGR2GRAY)
print('Create image : Done')

# Count the frequency of unique values in numpy array
unique_cat, counts_cat = np.unique(gray_cat, return_counts=True)
unique_dog, counts_dog = np.unique(gray_dog, return_counts=True)
unique_elements1, counts_elements1 = np.unique(gray_img1, return_counts=True)
unique_elements2, counts_elements2 = np.unique(gray_img2, return_counts=True)
unique_elements3, counts_elements3 = np.unique(gray_img3, return_counts=True)
unique_elements4, counts_elements4 = np.unique(gray_img4, return_counts=True)
unique_elements5, counts_elements5 = np.unique(gray_img5, return_counts=True)
unique_elements6, counts_elements6 = np.unique(gray_img6, return_counts=True)
print('Count the frequency of unique values in numpy array : DONE ')


Length = [len(counts_cat) , len(counts_dog) , len(counts_elements1) , len(counts_elements2)
    , len(counts_elements3), len(counts_elements4), len(counts_elements5), len(counts_elements6)]

#print(Length)
#print(max(Length))

ma_len = max(Length)

def AddValueisZero(arr , max_len): # arr vs max length
    if len(arr) < max_len:
        len_zero = max_len - len(arr)
        value_zero = np.zeros(len_zero, dtype=int)
        cre_arr = np.append(arr, value_zero)
        return cre_arr
    return arr
    pass

count_cat = AddValueisZero(counts_cat,ma_len)
count_dog = AddValueisZero(counts_dog,ma_len)
count_unipue1 = AddValueisZero(counts_elements1,ma_len)
count_unipue2 = AddValueisZero(counts_elements2,ma_len)
count_unipue3 = AddValueisZero(counts_elements3,ma_len)
count_unipue4 = AddValueisZero(counts_elements4,ma_len)
count_unipue5 = AddValueisZero(counts_elements5,ma_len)
count_unipue6 = AddValueisZero(counts_elements6,ma_len)

#process Histogram of object
His_cat = np.float32(count_cat)/256.0 # tính giá trị đặc trưng của ảnh
His_dog = np.float32(count_dog)/256.0
His_img1 = np.float32(count_unipue1)/256.0
His_img2 = np.float32(count_unipue2)/256.0
His_img3 = np.float32(count_unipue3)/256.0
His_img4 = np.float32(count_unipue4)/256.0
His_img5 = np.float32(count_unipue5)/256.0
His_img6 = np.float32(count_unipue6)/256.0

# process image cat , distance cosine basic :
dice_cat1 = round((1-np.sum(abs(His_img1 * His_cat)) / math.sqrt(np.sum(His_img1*His_img1) * np.sum(His_cat*His_cat))),3)
dice_cat2 = round((1-np.sum(abs(His_img2 * His_cat)) / math.sqrt(np.sum(His_img2*His_img2) * np.sum(His_cat*His_cat))),3)
dice_cat3 = round((1-np.sum(abs(His_img3 * His_cat)) / math.sqrt(np.sum(His_img3*His_img3) * np.sum(His_cat*His_cat))),3)
dice_cat4 = round((1-np.sum(abs(His_img4 * His_cat)) / math.sqrt(np.sum(His_img4*His_img4) * np.sum(His_cat*His_cat))),3)
dice_cat5 = round((1-np.sum(abs(His_img5 * His_cat)) / math.sqrt(np.sum(His_img5*His_img5) * np.sum(His_cat*His_cat))),3)
dice_cat6 = round((1-np.sum(abs(His_img6 * His_cat)) / math.sqrt(np.sum(His_img6*His_img6) * np.sum(His_cat*His_cat))),3)

# Chart distance cat photos.
Dice_cat = [dice_cat1,dice_cat2,dice_cat3,dice_cat4,dice_cat5,dice_cat6]

# process image dog , distance cosine basic :
dice_dog1 = round((1-np.sum(abs(His_img1 * His_dog)) / math.sqrt(np.sum(His_img1*His_img1) * np.sum(His_dog*His_dog))),3)
dice_dog2 = round((1-np.sum(abs(His_img2 * His_dog)) / math.sqrt(np.sum(His_img2*His_img2) * np.sum(His_dog*His_dog))),3)
dice_dog3 = round((1-np.sum(abs(His_img3 * His_dog)) / math.sqrt(np.sum(His_img3*His_img3) * np.sum(His_dog*His_dog))),3)
dice_dog4 = round((1-np.sum(abs(His_img4 * His_dog)) / math.sqrt(np.sum(His_img4*His_img4) * np.sum(His_dog*His_dog))),3)
dice_dog5 = round((1-np.sum(abs(His_img5 * His_dog)) / math.sqrt(np.sum(His_img5*His_img5) * np.sum(His_dog*His_dog))),3)
dice_dog6 = round((1-np.sum(abs(His_img6 * His_dog)) / math.sqrt(np.sum(His_img6*His_img6) * np.sum(His_dog*His_dog))),3)

#Chart distance dog photos.
Dice_dog = [dice_dog1,dice_dog2,dice_dog3,dice_dog4,dice_dog5,dice_dog6]

#Display Chart distance cat and dog
print('Chart Cat \n',Dice_cat)
print('Chart Dog \n',Dice_dog)

mean_Dcat = round(np.sum(Dice_cat)/len(Dice_cat),6)
mean_Ddog = round(np.sum(Dice_dog)/len(Dice_dog),6)

if mean_Dcat < mean_Ddog:
    cv2.putText(cat,'This is cat',(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow('This is cat',cat)
    cv2.waitKey()
else:
    cv2.putText(dog,'This is cat',(10, 25),cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
    cv2.imshow('This is cat', dog)
    cv2.waitKey()


cv2.destroyAllWindows()