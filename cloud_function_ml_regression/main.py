def main (data,context):
    
    processes= [
        "creating_dataset.py",
        "processing_dataset.py",
        "ml_regression.py",
        "remove_files.py"
        ] 

    for p in processes:
        print(p)
        exec(open(p).read())

if __name__ == "__main__":
  
    main('data','context')