import os
import cv2

class Loader:
    def __init__(self, *args):
        self.dir_paths = args
        self.resize = (640,480)
        self.final_output = self.process()

    def merge_and_randomize(self,input):
        output = []

        for y,dirs in enumerate(input):
            for j,img in enumerate(dirs):
                #print(f"{j}. x || {y}. y")
                output.append([img,y])
                
        return output

    def process(self):
        output = []
        for i,folder_path in enumerate(self.dir_paths):
            output.append([])
            if os.path.isdir(folder_path):
                for filename in os.listdir(folder_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        img = cv2.imread(os.path.join(folder_path, filename))

                        # ořezávání

                        img = cv2.resize(img, self.resize)
                        output[i].append(img)

                if output[i]:
                    #print(f"From dir {i} loaded {len(output[i])} images.")
                    print()
                else:
                    print("No images found in the directory.")

            else:
                print("Invalid directory path.")

        output = self.merge_and_randomize(output)
        #print(f"output len: {len(output)}")
        #print(output)
        return output
    
    def get(self,index):
        return self.final_output[index]

    



if __name__ == "__main__":
    ld = Loader("dataset/lomy/stepnylom_jpg","dataset/lomy/tvarnylom_jpg")
    print(ld.get(6))

