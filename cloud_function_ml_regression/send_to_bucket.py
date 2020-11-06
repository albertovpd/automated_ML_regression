import pandas as pd
from google.cloud import storage
import os
from dotenv import load_dotenv
load_dotenv()

results_list=[
    "results-features_evolution-append.csv",
"results-inferences-overwrite.csv",
 "results-ranking_of_features-overwrite.csv",
 "results-weekly_rmse-append.csv"
 ]

for r in results_list:
    df=pd.read_csv("../tmp/{}".format(r))
    df.to_csv('gs://{}/{}.csv'.format(os.getenv("bucket_in_gcs") ,r))
    print(r," sent to bucket")
