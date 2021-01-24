import pandas as pd
import numpy as np
from datetime import timedelta, datetime

from sklearn import preprocessing
from sklearn import decomposition 
from scipy import stats
from sklearn.covariance import EllipticEnvelope

import matplotlib.pyplot as plt
import matplotlib.font_manager
from my_functions import outliers_graph


dataset=pd.read_csv("../tmp/dataset_final.csv",index_col=[0])

# # outliers removal
# without_date= list(dataset.columns)
# without_date.remove("date")
# for c in without_date:
#     q_low= dataset[c].quantile(0.05)
#     q_high= dataset[c].quantile(0.95)
#     df= dataset[(dataset[c]<q_high) & (dataset[c]> q_low)]
# removed=int(100*(dataset.shape[0]-df.shape[0])/df.shape[0])



# sliding time moving everything 4 weeks ahead, to have an "x axis" from which infer my target
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

outliers_dect= df.drop(columns=["date","unemployment"])
# 2D reduction
pca=decomposition.PCA()
pca.n_components=2
pca_data=pca.fit_transform(outliers_dect)
pca_data=pd.DataFrame(pca_data)
pca_data.rename(columns={0:"a",1:"b"}, inplace=True)

# outliers detection.
outlier_method = EllipticEnvelope().fit(pca_data)
scores_pred = outlier_method.decision_function(pca_data)
threshold = stats.scoreatpercentile(scores_pred, 20) # remove just the heaviest outliers

# outliers removal
df["outliers_score"]=scores_pred
df =df[df["outliers_score"]>=threshold]
print("-2- df processed without outliers shape: ",df.shape)

# Plot outliers.
plot_min=int(min(list(pca_data.min())))
plot_max=int(max(list(pca_data.max())))
blacks = len(scores_pred)-len(scores_pred[scores_pred<threshold])

outliers_graph(pca_data, outlier_method, blacks, threshold, plot_min, plot_max)
#------------------------


df.to_csv("../tmp/dataset_final_processed.csv") 
print("2: dataset without outliers sent to /tmp")

