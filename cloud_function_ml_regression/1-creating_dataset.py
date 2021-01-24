import os
import pandas as pd
from my_functions import creating_dataset
from datetime import datetime, date

from dotenv import load_dotenv
load_dotenv()

# working remote => in .env, save the path without "", like:    df_google   =   ./folder/whatever
# working on the cloud => the key of your token is the bucket:  df_google   =   gs://<yourbucket>/<yourcsv>)

df_google= pd.read_csv("../tmp/data_pytrends.csv")
df_google.sort_values(by=["date"],inplace=True)
print("1: google dataset loaded")

df_economical=pd.read_csv("../tmp/dashboard_spanish_news_economical.csv.gz",compression='gzip', header=0, quotechar='"', error_bad_lines=False)
df_economical.sort_values(by=["Date"],inplace=True)

df_political=pd.read_csv("../tmp/dashboard_spanish_news_political.csv.gz",compression='gzip', header=0, quotechar='"', error_bad_lines=False)
df_political.sort_values(by=["Date"],inplace=True)

df_social=pd.read_csv("../tmp/dashboard_spanish_news_social.csv.gz",compression='gzip', header=0, quotechar='"', error_bad_lines=False)
df_social.sort_values(by=["Date"],inplace=True)

print("1: gdelt datasets loaded")

# creating df
df_google_dates=pd.DataFrame()

# creating the Date column in new dataset
df_google_dates["date"]=list(set(df_google["date"]))
df_google_dates["date"]=pd.to_datetime(df_google_dates["date"])
df_google_dates.sort_values(by=["date"],inplace=True)

# Creating the new columns. Trend index with the name of the corresponding keyword
keyword_list=list(set(df_google["keyword"]))
keyword_list.sort()
for k in keyword_list:
    df_google_dates[k]=df_google[(df_google['keyword'] == k)]["trend_index"].tolist()

df_google_dates.rename(columns={'desempleo':"unemployment"}, inplace=True)

# News datasets
dfp = creating_dataset(df_political,"political")
dfs = creating_dataset(df_social,"social")
dfe = creating_dataset(df_economical,"economical")

# Merging datasets
datasets = [ dfp, dfe, dfs,df_google_dates]
date2 = []

for d in datasets:
    date2. append(d.date.max()) # get the later date of every dataset
    
date2 = max(date2) 

# creating final dataset with everything
date1 = '2019-01-01' 
mydates = pd.date_range(date1, date2, freq="W").tolist()
df_final=pd.DataFrame()
df_final["date"]=mydates
df_final['date']=pd.to_datetime(df_final["date"])

for d in datasets:
    df_final=df_final.merge(d,how='left', left_on="date", right_on="date",suffixes=["_1","_2"])

df_final=df_final.fillna(0)
df_final.to_csv("../tmp/dataset_final.csv")
#df_final.to_csv("gs://--yourbucket--/step1-df_merged.csv")   #<==================================== 2
print("1: raw dataset created")


