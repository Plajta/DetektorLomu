import os
import cv2
import random
import numpy as np

class Loader:
    def __init__(self, *args):
        self.dir_paths = args
        self.resize = (640,480)
        self.output = []
        self.training = []
        self.testing = []
        self.final_output = self.process()

    def merge_and_randomize(self,input):
        output = []

        for y,dirs in enumerate(input):
            for j,img in enumerate(dirs):
                #print(f"{j}. x || {y}. y")
                output.append([img,y])
                
        return output
    
    def resizing(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        left_corner = img[round(len(img)*0.88):round(len(img)),0:round(len(img[0])*0.15)]
        right_corner = img[round(len(img)*0.88):round(len(img)),round(len(img[0])*0.85):round(len(img[0]))]
        if (np.sum((left_corner<20)) >= 500 or np.sum((right_corner<20)) >= 500): img = img[0:round(len(img)*0.88),:]
        for num, line in enumerate(img[round(len(img)*0.8):len(img)]):
            if sum(line>=200)>50:
                img = img[0:round(len(img)*0.8)+num-30]
                break
        # ořezávání
        img = cv2.resize(img, self.resize)
        return img

    def process(self):
        for i,folder_path in enumerate(self.dir_paths):
            self.output.append([])
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        img = cv2.imread(os.path.join(folder_path, filename))
                        img = self.resizing(img)
                        
                        self.output[i].append(img)

                if self.output[i]:
                    #print(f"From dir {i} loaded {len(output[i])} images.")
                    print()
                else:
                    print("No images found in the directory.")

            else:
                print("Invalid directory path.")

        #print(f"output len: {len(self.output)}")
        #print(self.output)
        return self.merge_and_randomize(self.output)
    


    def randomize(self):
        random.shuffle(self.final_output)
    
    def get(self,index,which_dataset=0):
        match which_dataset:
            case 0:
                return self.final_output[index]
            case 1:
                return self.training[index]
            case 2:
                return self.testing[index]

    def get_array(self,which_dataset=0):
        match which_dataset:
            case 0:
                return self.final_output
            case 1:
                return self.training
            case 2:
                return self.testing        
    
    def get_length(self,which_dataset=0):
        match which_dataset:
            case 0:
                return len(self.final_output)
            case 1:
                return len(self.training)
            case 2:
                return len(self.testing)
    
    def generate_dataset(self,number_of_articles):
        print("generate")
        from_number = round(number_of_articles/len(self.output))
        print(from_number)
        print(len(self.output))
        print(len(self.output[0]))

        for i in range(from_number):
            for y,j in enumerate(self.output):
                self.training.append([j[i],y])

        random.shuffle(self.training)

        print(range(from_number,len(self.final_output[0])))

        for y,dir in enumerate(self.output):
            for i in range(from_number,len(dir)):
                self.testing.append([dir[i],y])
        random.shuffle(self.testing)

    def get_some_bitches(self):
        print("you? never")

if __name__ == "__main__":
    ld = Loader("dataset/lomy/stepnylom_jpg","dataset/lomy/tvarnylom_jpg")
    print(ld.get(6))
    ld.generate_dataset(400)
    print(ld.get_length(0))
    print(ld.get_length(1))
    print(ld.get_length(2))
    cv2.imshow('Grayscale Image', ld.get(2)[0])
    print(ld.get_length())
    ld.randomize()
    jed = 0
    nul = 0
    for i in ld.get_array(1):
        if i[1] == 1:
            jed+=1
        else:
            nul+=1
        print(i[1])
    print(f"jed: {jed} nul: {nul}")
    cv2.imshow('Example Image', ld.get(5,1)[0])

    for set in ld.get_array(1):

        cv2.imshow('Image', set[0])


        key = cv2.waitKey(0)

        if key == 13:
            continue

        else:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


