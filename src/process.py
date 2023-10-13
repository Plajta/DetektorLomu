import os
import cv2
import numpy as np

class Loader:
    def __init__(self, *args):
        self.dir_paths = args

    def do(self):
        output = []
        for i,folder_path in enumerate(self.dir_paths):
            output.append([])
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        img = cv2.imread(os.path.join(folder_path, filename))
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

