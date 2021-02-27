import pandas as pd
import numpy as np
from datetime import timedelta, datetime, date

from sklearn import preprocessing
from sklearn import decomposition 
from scipy import stats
#from sklearn.covariance import EllipticEnvelope

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.font_manager
from my_functions import plot_dimensions   #outliers_graph


dataset=pd.read_csv("../tmp/dataset_final.csv",index_col=[0])

# SLIDING
# moving everything 4 weeks ahead, to have an "x axis" from which infer my target
dataset["date"]=pd.to_datetime(dataset["date"])
df_unemployment=pd.DataFrame(dataset[["date", "unemployment"]])
dataset.drop(columns="unemployment", inplace=True)

dataset["date"]=dataset["date"].apply(lambda x: x+timedelta(weeks=4))

# now let's merge again. outter to get all from both datasets
df=pd.merge(dataset,df_unemployment,how='outer', on=["date"],suffixes=(None,None))
df.sort_values(by=["date"],inplace=True)
df=df.fillna(0) 
df.reset_index(drop=True,inplace=True)
# lets remove the first 4 rows. We don't have data in our X, just in our target, so it's redundant
df.drop([0,1,2,3], inplace=True)
df.reset_index(drop=True, inplace=True) # very important


# REMOVING OUTLIERS
column_list=list(df.columns)
column_list.remove("date") # it is no sense processing this column

# using zscore for each column
df_outliers=df[column_list]
z_scores = stats.zscore(df_outliers)
abs_z_scores = np.abs(z_scores)

# we should really work with zscore <= +-3 to stick to the math, but i'll loose a lot of data
filtered_entries = (abs_z_scores < 6).all(axis=1)
df_without_outliers = df_outliers[filtered_entries]
percentaje = round( 100*(1 - df_without_outliers.shape[0]/df_outliers.shape[0]))
print("2: {} % of rows removed".format(percentaje))

# for dashboarding the weekly removed percentaje
# google trends data varies a lot every week, so maybe it is worth check it out
today=datetime.now().date()
cleaning_results= pd.DataFrame({"date":[today], "percentaje_removed":[percentaje]})
cleaning_results.to_csv("../tmp/cleaning_results.csv")   # <=========================================

# ------------------
# clean dataset for ML
df_date=pd.DataFrame() 
df_date["date"]=df.date

# now we add it by the index
dataset=df_without_outliers.merge(df_date,how='left', left_index=True, right_index=True)
dataset.to_csv("../tmp/dataset_final_processed.csv")



# Plotting outliers
plot_dimensions(df_outliers[~filtered_entries],df_outliers[filtered_entries],2)
plot_dimensions(df_outliers[~filtered_entries],df_outliers[filtered_entries],3)

print("2: plots sent to tmp folder")

print("2: dataset without outliers sent to /tmp")

