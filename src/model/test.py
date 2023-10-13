import cv2 
from process import Loader

ld = Loader("dataset/lomy/stepnylom_jpg","dataset/lomy/tvarnylom_jpg")
ld.randomize()
print(ld.get_length())
for i in range(20):
    
    img, label = ld.get(i)
    cv2.imshow(img)
