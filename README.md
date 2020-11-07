# Automated ML within a Cloud Function.

![alt](output/under_catstruction.jpeg "gatito de la costrucción reza por mí")
Yeah, in development.

I have at hand an automated ETL ingesting data periodically, it measures many keywords allocated in different BigQuery tables, from different datasets in different regions (*Google Cloud Data Engineer fun here. If you know this pain, pray for me*).  

After processed, the different sources will be merged into a single dataset, with every keyword in a column with values for each week. 

With that, I'll modify the dataset as I need, to infer some of the columns with 4 weeks in advance. In the jupyter notebooks shown here, I explained as detailed as I could the procedure.

I'm having results already and they are surprising. It works: The result is not good at the moment, nevertheless it is way better than expected.

Now I need to refactorize code as much as I can to insert the regression within a Cloud Function with 4GB of RAM and 60 seconds of timeout. If it works, I'm happy regardless the performance.
Myself from the future will improve the script, and get better metrics through model selection, tunning of hyperparameters, etc, etc. Right now the priority is make my baby work in my automated pipeline. Enhancement is secondary at the moment.


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


### Schedulers
<details>
  <summary>Click to expand</summary>

The ETL with which I'm feeding my project is weekly updated on Mondays. I have no rush so I'll run pipelines on Tuesdays.

- Cloud Function reading tables from BigQuery and loading into Cloud Storage bucket (EEUU) => 0 1 * * 2 CET (Belgium). Topic => tuesdays-reading-bq

- Cloud Function reading from Cloud Storage, applying my ML regression and delivering data again to Storage => 0 2 * * 2 CET (Belgium). Topic => reading_from_cs



</details>

------------------------------------


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

-------------------------------------

<details>
  <summary>Click to expand</summary>

## Schedulers
<details>
  <summary>Under construction</summary>

- Cloud function loading Gdelt data from BigQuery to Cloud Storage EEUU: 0 1 * * 2 (every Tuesday at 1:00). 

</details>

----------------------------------



------------------------------------

### Create the ML Cloud Function:

<details>
  <summary>Under construction</summary>

  - Don't forget to use your service account, the same than the feeding project => URL
  - It is possible to read from different buckets with the same Cloud Function, yeah.
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




