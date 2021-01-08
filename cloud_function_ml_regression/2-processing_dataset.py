import pandas as pd
from datetime import timedelta
import numpy as np

from sklearn import preprocessing
from sklearn import decomposition 
from scipy import stats
from sklearn.covariance import EllipticEnvelope

import matplotlib.pyplot as plt
import matplotlib.font_manager

from my_functions import outliers_graph


dataset=pd.read_csv("../tmp/dataset_final.csv",index_col=[0])

# sliding time
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
threshold = stats.scoreatpercentile(scores_pred, 15) # remove just the heaviest outliers

# outliers removal
df["outliers_score"]=scores_pred
df =df[df["outliers_score"]>=threshold]

# Plot outliers.
plot_min=int(min(list(pca_data.min())))
plot_max=int(max(list(pca_data.max())))
outliers_graph(pca_data, outlier_method, 150, threshold, plot_min, plot_max)

df.to_csv("../tmp/dataset_final_processed.csv") #<========================

print("dataset without outliers sent to /tmp")
