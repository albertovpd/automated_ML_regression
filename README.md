# Automated ML within a Cloud Function.

![alt](output/under_catstruction.jpeg "gatito de la costrucción reza por mí")
Yeah, in development.

I have at hand an automated ETL ingesting data periodically, it measures many keywords allocated in different BigQuery tables, from different datasets in different regions (*Google Cloud Data Engineer fun here. If you know this pain, pray for me*).  

After processed, the different sources will be merged into a single dataset, with every keyword in a column with values for each week. 

- Will I be able to foresee the value of a column 2 weeks in advance? I think so, but the result is not going to be great, because everything will be triggered from several *Cloud Functions*, which have 60 seconds of working time as maximum, and 4GB of RAM for loading and processing.

- Then, if you are already aware of the not-great performance, why are you doing it? Because, working with Cloud Functions is really cheap, a compulsory point for leisure project. Moreover, I believe is the cheapest way of automating a process in Google Cloud. It is worth playing with it, and if in the end the result is not suitable in any way, I will have learnt:
    - How to work with Cloud Functions for many purposes: Loading from BQ, Loading from many regions.
    - Feature engineering necessary to minimize RAM impact
    - Tunning and understanding the ML regression to minimize RAM impact 

Of course, the goal is automate and implement this to the pipeline, in order to display in a final dashboard everything + the ML part.

# What data I am using


<details>
  <summary>Click to expand</summary>

  A **Python** Cloud Function is requesting weekly from an API. The data is processed and loaded finally to several BigQuery tables for display in Data Studio. The Cloud Function, Cloud Storage bucket, Transfer and BigQuery dataset is in *EU region*. 

  There is a request from the *Gdelt* Project through BigQuery with **SQL**. Google has the *Gdelt thing* allocated in USA servers, so the retrieved information is stored in a dataset located in *USA*. The dataset has many tables, and they are also displayed in Data Studio.

  You can find everything in detail here:

  - Dashboard => https://datastudio.google.com/s/iFQxr4r9ocs
  - Repository => https://github.com/albertovpd/automated_etl_google_cloud-social_dashboard
  

</details>

---------------------------------------------


# The Data Engineering behind.

<details>
  <summary>Click to expand</summary>
  
### Load from BigQuery to Cloud Storage
<details>
  <summary>Click to expand</summary>

In *cloud_function_from_bq_to_storage.py* you will find the script, and the stackoverflow source where I found it.

Extras, configuration:

- Create a CF, name it and choose a processing capacity (study it before configuring the CF, you can have errors for not having enough capacity).
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

    - I included *os* in *requirements.txt*, but I it is not necessary.

</details>

------------------------------------

### Create the ML Cloud Function:

<details>
  <summary>Under construction</summary>
    - Ingest several csv from several *REGIONS*
    - In case it is not possible, I have 2 regions, *EUROPE, USA*. The automation of coping the data of one region to another is necessary. I would like to avoid cron jobs and the Cloud terminal if possible.

</details>

### Load the result again in Storage

### Load it in a BigQuery table

### Automate everything to append the weekly data, not everything constantly

This is going to be mental, several ideas but still figuring it out. Problem for myself from the future.


</details>

---------------------------------------------

# Exploring my chances

<details>
  <summary>Under construction</summary>

</details>

---------------------------------------------


# Final script with the regression


<details>
  <summary>Under construction</summary>

</details>

---------------------------------------------

# Final script with the regression. Prioritizing speed vs performance.


<details>
  <summary>Under construction</summary>

I am working within a Cloud Function, that means I have 60 seconds as maximum to finish the script and 4G of RAM for processing and loading. It looks like it is not the place for DNN precisely 

</details>

---------------------------------------------




