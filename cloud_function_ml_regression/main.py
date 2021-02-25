# i need to import libraries here to avoid conflicts
from datetime import datetime, timedelta
import numpy as np
from math import sqrt
<<<<<<< HEAD
import sklearn.metrics as metrics
=======
import sklearn.metrics as metrics 

>>>>>>> df25908adba494b4a60890544fc9c348528fce93
#from google.cloud import storage


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
        #,
        #"4-sending_pics_to_storage.py"
                 ] 

    for p in processes:
        exec(open(p).read())
        print("=>",p, "done")

if __name__ == "__main__":
  
    main('data','context')