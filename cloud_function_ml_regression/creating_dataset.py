import os
import pandas as pd
from my_functions import creating_dataset
from datetime import datetime, date

from dotenv import load_dotenv
load_dotenv()

# in .env, save the path without "", like: df_google = ./folder/whatever
df_google= pd.read_csv(os.getenv("df_google"))
print("google dataset loaded")

df_economical=pd.read_csv(os.getenv("df_economical"))
df_political=pd.read_csv(os.getenv("df_political"))
df_social=pd.read_csv(os.getenv("df_social"))
print("gdelt datasets loaded")

df_google.drop(columns='Unnamed: 0', inplace=True)

# creating df for google
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


# Using my function to create columns from keywords with Gdelt datasets
dfp = creating_dataset(df_political,"political")
dfs = creating_dataset(df_social,"social")
dfe = creating_dataset(df_economical,"economical")

# creating final dataset with everything
date1 = '2019-01-01'
date2 = datetime.now().date()
#date2="2020-10-18"
mydates = pd.date_range(date1, date2, freq="W").tolist()
df_final=pd.DataFrame()
df_final["date"]=mydates
df_final['date']=pd.to_datetime(df_final["date"])

datasets = [ dfp, dfe, dfs,df_google_dates]

for d in datasets:
    df_final=df_final.merge(d,how='left', left_on="date", right_on="date")

df_final=df_final.fillna(0)
df_final.to_csv("../tmp/dataset_final.csv")


print("raw dataset created")

