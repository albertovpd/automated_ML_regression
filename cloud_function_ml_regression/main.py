import datetime
import numpy as np
# Not defining datetime here will end in an error

def main (data,context):
    
    processes= [
        "creating_dataset.py",
        "processing_dataset.py",
        "ml_regression.py",
        "remove_files.py"
        ] 

    for p in processes:
        exec(open(p).read())
        print("=>",p, "done")

if __name__ == "__main__":
  
    main('data','context')