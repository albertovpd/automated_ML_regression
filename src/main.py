import os
from os import listdir
from os.path import isfile, join

def main (data,context):
    
    processes= ["loading_datasets.py","merging_datasets.py","ml_regression.py","remove_files.py"] 

    for p in processes:
        print(p)
        exec(open(p).read())

if __name__ == "__main__":
  
    main('data','context')