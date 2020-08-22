# automated_ML_in_production
What's the cheapest way of having a ML model deployed and working, self-evaluating its metrics with each automated batch ingestion and updating itself if necessary? I don't know yet, but I have an idea


# Introduction.


**The plan:**



# Cloud Functions

## 1. Periodically load from BigQuery to Cloud Storage.

From CS to BigQuery is really easy and user-friendly, using Transfer, in the BigQuery interface. I needed some StackOverflow to perform it, and this solution works really well: 

=> https://stackoverflow.com/questions/59687796/how-to-schedule-an-export-from-a-bigquery-table-to-cloud-storage

Configuration:

- Give to it a nice name and a processing capacity (study it before configuring the CF, you can have errors for not having enough capacity).
- Configure it with *PUB/SUB*, to activate it through Cloud Scheduler.
- In Advanced, select *Environmental Variables*:
    - Write all of them, keys and values, without declaring *str* type. I mean, without the quotation marks **" "**.
    - In your CF script, replace:

        project_name = "YOUR_PROJECT_ID" 
        bucket_name = "YOUR_BUCKET" 
        dataset_name = "YOUR_DATASET" 
        table_name = "YOUR_TABLE" 

    - By:
    
        project_name = os.getenv("YOUR_PROJECT_ID") 
        bucket_name = os.getenv("YOUR_BUCKET") 
        dataset_name = os.getenv("YOUR_DATASET") 
        table_name = os.getenv("YOUR_TABLE") 

    - I included *os* in *requirements.txt*, but I think is not necessary, in the end, we are activating a small and fast Linux machine.