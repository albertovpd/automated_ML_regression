import os
from google.cloud import bigquery

def bigquery_request(self, request):
    # Replace these values according to your project

    project_name = os.getenv("project_name") 
    bucket_name = os.getenv("bucket_name") 
    dataset_name = os.getenv("dataset_name") 

    table_names = ["dashboard_spanish_news_economical","dashboard_spanish_news_political","dashboard_spanish_news_social"] 

    for t in table_names:
        destination_uri = "gs://{}/{}.csv.gz".format(bucket_name, t)


        bq_client = bigquery.Client(project=project_name)

        dataset = bq_client.dataset(dataset_name, project=project_name)
        table_to_export = dataset.table(t)

        job_config = bigquery.job.ExtractJobConfig()
        job_config.compression = bigquery.Compression.GZIP

        extract_job = bq_client.extract_table(
            table_to_export,
            destination_uri,
            # Location must match that of the source table.
            location="US",
            job_config=job_config
            )

    return "Job with ID {} started exporting data from {} to {}".format(extract_job.job_id, dataset_name,bucket_name)
