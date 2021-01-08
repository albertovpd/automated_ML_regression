# i need to import libraries here to avoid conflicts
from datetime import datetime, timedelta
import numpy as np

def main (data,context):

    '''
    The problem I want to solve have been separated into many steps, more than necessary
    for good development practices, but it's easier for me to debug in this way. Myself 
    from the future will take care of refactoring, once the deployment works propertly. 
    '''
    
    processes= [
        "1-creating_dataset.py",
        "2-processing_dataset.py",
        "3-ml_regression.py"
        #,"4-sending_to_cs.py"
                ] 

    for p in processes:
        exec(open(p).read())
        print("=>",p, "done")

if __name__ == "__main__":
  
    main('data','context')