import os
import cv2
import numpy as np

class Loader:
    def __init__(self, *args):
        self.dir_paths = args
        self.resize = (640,480)

    def do(self):
        output = []
        for i,folder_path in enumerate(self.dir_paths):
            output.append([])
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        img = cv2.imread(os.path.join(folder_path, filename))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        left_corner = img[round(len(img)*0.88):round(len(img)),0:round(len(img[0])*0.15)]
                        right_corner = img[round(len(img)*0.88):round(len(img)),round(len(img[0])*0.85):round(len(img[0]))]
                        if (np.sum((left_corner<20)) >= 500 or np.sum((right_corner<20)) >= 500): img = img[0:round(len(img)*0.88),:]
                        for num, line in enumerate(img):
                            if sum(line>=200)>50 and num>len(img)*0.8:
                                img = img[0:num-30]
                                break
                        img = cv2.resize(img, self.resize)
                        cv2.imwrite(f"/home/vasek/plajta/dataset/out/{filename}",img)
                        output[i].append(img)

                if output[i]:
                    print(f"From dir {i} loaded {len(output[i])} images.")
                else:
                    print("No images found in the directory.")
            else:
                print("Invalid directory path.")
            
        #print(output)



ld = Loader("dataset/lomy/stepnylom_jpg","dataset/lomy/tvarnylom_jpg")

ld.do()

