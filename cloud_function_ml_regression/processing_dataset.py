import pandas as pd
from datetime import timedelta


dataset=pd.read_csv("../tmp/dataset_final.csv")

dataset.drop(columns='Unnamed: 0', inplace=True)
# Unemployment, which is how much people searches "desempleo" in Spain, will be my target to infere
dataset.rename(columns={'desempleo_y':"unemployment"}, inplace=True)

# Sliding target column
dataset["date"]=pd.to_datetime(dataset["date"])
df_unemployment=pd.DataFrame(dataset[["date", "unemployment"]])
dataset.drop(columns="unemployment", inplace=True)

dataset["date"]=dataset["date"].apply(lambda x: x+timedelta(weeks=4))

# now let's merge again
df=pd.merge(dataset,df_unemployment,how='outer', on=["date"],suffixes=(None,None))
df.sort_values(by=["date"],inplace=True)
df=df.fillna(0)

df.reset_index(drop=True,inplace=True)
# lets remove the first 3 rows, they are redundant
df.drop([0,1,2,3], inplace=True)

df.to_csv("../tmp/dataset_final_processed.csv")
