import datetime
import numpy as np

def main (data,context):

    '''
    The problem I want to solve have been separated into many steps, more than necessary
    for good development practices, but it's easier for me to debug in this way. Myself 
    from the future will take care of refactorizing, once the deployment works propertly. 
    '''
    
    processes= [
        "creating_dataset.py",
        "processing_dataset.py",
        "ml_regression.py"#,
        #"sending_to_bucket.py",
        #"remove_files.py"
        ] 

    for p in processes:
        exec(open(p).read())
        print("=>",p, "done")

if __name__ == "__main__":
  
    main('data','context')