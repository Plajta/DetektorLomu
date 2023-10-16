import sys, getopt, os, random, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from cv2 import imread

DEFAULT_DIRECTORIES = ["dataset/lomy/stepnylom_jpg","dataset/lomy/tvarnylom_jpg"]
BENCHMARK_PATH = "benchmark"

def print_help():
    print("CVUTHack Benchmark by Plajta <3")
    print("Usage: benchmark.py [command]")
    print(f"Default directories: {DEFAULT_DIRECTORIES}")
    print("\nCommands:")
    print("\trd\t\t\tRegenerates the test dataset, without other arguments takes 20 images from all default directiories")
    print("\trd [num]\t\tRegenerates the test dataset with [num] images from all default directories")
    print("\trd [num] [dir]...\tRegenerates the test dataset with [num] images from all specified directories")
    print("\ttest\t\t\tWell...runs the test")
    print("\thelp\t\t\tPrints this text")

def regenerate_benchmark_dataset(img_num,*paths):
    # If the benchmark directory doesn't exist make it, otherwise empty it
    if os.path.exists(BENCHMARK_PATH): shutil.rmtree(BENCHMARK_PATH)
    os.mkdir(BENCHMARK_PATH)
    
    for i,path in enumerate(paths):
        if os.path.isdir(path):
            new_path = os.path.join(BENCHMARK_PATH,str(i))
            os.mkdir(new_path)
            list_of_files = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name))]
            if len(list_of_files) >= img_num:
                randomized = random.sample(list_of_files,img_num)
                for element in randomized: shutil.copy(os.path.join(path,element),os.path.join(new_path,element))
        else:
            print(f"{path} is not a directory!")

def start_test():
    from model.processing import Loader
    from model.inference import infer_CNN
    loader = Loader()
    errors = 0
    for root, dirs, files in os.walk(BENCHMARK_PATH):
        for name in files:
            resized_img = loader.resizing(imread(os.path.join(root,name)))
            result = infer_CNN(resized_img, "src/model/saved/CNN/cnn_best.keras") # TODO Make it configurable
            if result != int(root[-1]):
                errors += 1
    print(f"Number of errors: {errors}")

def main(args):
    if not len(args):
        print_help()
    elif len(args) == 1:
        match args[0]:
            case "help":
                print_help()
            case "rd":
                regenerate_benchmark_dataset(20,*DEFAULT_DIRECTORIES)
            case "test":
                start_test()
    elif len(args) == 2:
        if args[0] == "rd":
            regenerate_benchmark_dataset(int(args[1]),*DEFAULT_DIRECTORIES)
    else:
        if args[0] == "rd":
            regenerate_benchmark_dataset(int(args[1]),*args[2:])

if __name__ == "__main__":
   main(sys.argv[1:])