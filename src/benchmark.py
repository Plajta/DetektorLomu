import sys, getopt, os, random, shutil

DEFAULT_DIRECTORIES = ["dataset/lomy/stepnylom_jpg","dataset/lomy/tvarnylom_jpg"]

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
    benchmark_path = "benchmark"
    # If the benchmark directory doesn't exist make it, otherwise empty it
    if os.path.exists(benchmark_path): shutil.rmtree(benchmark_path)
    os.mkdir(benchmark_path)
    
    for i,path in enumerate(paths):
        if os.path.isdir(path):
            new_path = os.path.join(benchmark_path,str(i))
            os.mkdir(new_path)
            list_of_files = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path,name))]
            if len(list_of_files) >= img_num:
                randomized = random.sample(list_of_files,img_num)
                for element in randomized: shutil.copy(os.path.join(path,element),os.path.join(new_path,element))
        else:
            print(f"{path} is not a directory!")

def start_test():
    # from model.processing import Loader
    print(os.listdir("benchmark"))
    # loader = Loader()

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