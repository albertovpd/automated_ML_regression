

## Automated ML regression within a Cloud Function to infer unemployment searches on Google, in Spain:

----------------------------
----------------------------

![alt](output/automated_ml_regression.gif)

### **Dashboard** => https://datastudio.google.com/s/iGEf2faYhUE

-------------------------------------
-------------------------------------

# Introduction.

<details>
  <summary>Click to expand</summary>
  

Taking advantage of this project ( **https://github.com/albertovpd/automated_etl_google_cloud-social_dashboard** ), I am using the gathered data to feed a ML model with which inferring unemployment searches on Google, in Spain.

This project consists on:

+ Cloud Function A: Loads data from BigQuery tables to Cloud Storage, both in EEUU region. This tables contain requested and filtered info from the Gdelt Project, to analyse online news media in Spain (news section in the automated ETL link).

- Cloud Function B: 
  - Reads the data of Cloud Function A, and other data from a bucket in EU. This bucket contains requested info from Google Trends in Spain (Google searches section in the automated ETL link).
  - Merges datasets with different length and dates.
  - Processes them and creates a column and score for each keyword.
  - Normalizes the final dataset.
  - Associate date with index, but dates are not in the game, so a time series problem was turned into a linear regression one. Check it out the full script explanation here.
  - Performs a Recursive Feature Elimination to select the best 20 features of 130 I have to play with.
  - Apply a linear regression to infer my keyword, in this case, unemployment. 
  - Loads the results in a Cloud Storage bucket.

+ Both Cloud Functions are triggered by Pub/Sub and Scheduler. Scripts can be found here.

+ Weekly loaded to BigQuery tables with Transfer. Some results appended to the existing tables and some overwritten. 

+ Plot the BigQuery tables.

</details>

------------------------------------




# The Data Engineering behind.

<details>
  <summary>Click to expand</summary>

The processes involved are shown in *Introduction*.

### Schedulers


The ETL with which I'm feeding my project is weekly updated on Mondays. I have no rush so I'll run pipelines on Tuesdays.

- Cloud Function reading tables from BigQuery and loading into Cloud Storage bucket (EEUU) => 0 1 * * 2 CET (Belgium). Topic => tuesdays-reading-bq
- Cloud Function reading from Cloud Storage, applying my ML regression and delivering data again to Storage (EEUU) => 0 2 * * 2 CET (Belgium). Topic => reading_from_cs

- Transfer ml_regression-unemployment_inferences => Every Tue at 04:30:00 Europe/Paris => Field delimiter: ,  => Header rows: 1
- Transfer ml_regression-evolution_features => Every Tue at 04:30:00 Europe/Paris => Field delimiter: ,  => Header rows: 1
- Transfer ml_regression-weekly_score => Every Tue at 04:30:00 Europe/Paris => Field delimiter: ,  => Header rows: 1


### Creating tables in BigQuery

Now that my Cloud Function delivered the results to Cloud Storage, I need to load the data into a new dataset in BigQuery (based in EEUU, as my bucket).

- Create tables for every csv delivered in CS
- Advanced => Header rows to skip:1, comma separated

### Configure Transfers

Once the tables are created is necessary to configure Transfer for weekly automated updates of the tables. Beware of timing, you need to wait more or less 1 hour from loading to Storage, if don't, Transfer won't detect new files.

### Load from BigQuery to Cloud Storage


In **cloud_function_from_bq_to_storage.py** you will find the script, and the *stack overflow* source where I found it.

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


</details>

---------------

# ML explanation. 

<details>
  <summary>Click to expand</summary>

Here **https://github.com/albertovpd/automated_ML_regression/tree/master/cloud_function_ml_regression** you will find the Cloud Function script.


The processing part of the ML Cloud Function is explained in this jupyter in detail => **https://github.com/albertovpd/automated_ML_regression/blob/master/script_explained.ipynb**



My goal was to have the automated pipeline working. Myself from the future will refactorize the code, perform a better feature selection and optimise everything... Or not, in the end this is a leisure project and the goal is learning. I know now how to do it an also, how to do it way better. Goal accomplished.

</details>

------------------------------

# Achievements.

<details>
  <summary>Click to expand</summary>

The goal was to automate a ML model within a Cloud Function and infer data from a previous ETL. A Cloud Function has 4GB of RAM and 60 seconds of timeout, I fet it like a challenge.

It was the first upgrade of my ETL. It has a lot of room for improvement, but it runs and delivers somehow coherent results, I am happy for that.  

</details>


# Improvements

<details>
  <summary>Click to expand</summary>

- Cloud Function with ML regression:

The code is redundant. It requires refactorization, a lot.

- The ML part:

I would like to work with a set of fast models and implement them in the Cloud Function, so maybe every week a defferent model wins. Also, split my data into train/validation/test instead of k-folds for validation.

Instead of performing a multiple linear regression, I want to perform a script in which running a linear regression, multiple times with different targets (it's going to be the same, but it's going to be coded by myself. I know how to do it already and it will be easy and elegant).

</details>
