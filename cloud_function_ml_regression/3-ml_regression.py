import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
#from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import matplotlib.font_manager

from my_functions import variance_threshold_selector

df=pd.read_csv("../tmp/dataset_final_processed.csv",index_col=[0])

# Low variance removal
low_v = df.drop(columns=["date","unemployment","outliers_score" ])
X_raw= variance_threshold_selector(low_v, 5) # removing values than vary less than 5%

removed=[]
for c in low_v.columns:
    if c not in X_raw.columns:
        #print(c)
        removed.append(c)
removed=pd.DataFrame(removed, columns =["Removed_columns"])
removed.to_csv("../tmp/removed_features.csv")   #<==========================

# Normalization
normalizer = preprocessing.MinMaxScaler()
X = pd.DataFrame(normalizer.fit_transform(X_raw))

X.columns= X_raw.columns
target=df["unemployment"]

# all but last 4 rows
X_train=X.iloc[:-4]
target_train=target.iloc[:-4]
# last 4 rows, which are the ones I have no data and I want to infer
X_test=X.iloc[-4:]
target_test=target.iloc[-4:]


# ---------REGRESSION----------------
regression = Lasso(alpha=0.1)
#regression = LinearRegression()

min_number_features =  df.shape[0]//10
rfecv = RFECV(estimator=regression,
              step=1, 
              min_features_to_select=min_number_features, 
              cv=KFold(n_splits=10, shuffle=True, random_state=42), 
              scoring='neg_mean_squared_error')
rfecv.fit(X_train, target_train)
regression.fit(X_train,target_train)
print("training done")

#date
today=datetime.now()

# selecting features
number_features=pd.DataFrame()
number_features["number_columns"]=pd.Series(rfecv.n_features_)
number_features["date"]= pd.to_datetime(today)
number_features.to_csv("../tmp/number_features.csv") # <=====================

# plot features vs performance
plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=34 , pad=20)
plt.xlabel('Number of features selected', fontsize=28, labelpad=20)
plt.ylabel('Inner algorithm score', fontsize=28, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

plt.savefig("../tmp/RFE_columns.jpg")   # <==================================
print("feature selection done")

#-----------HASTA AQUÍ------------------ // TODO hasta aquí 

#####################
#       RESULTS     #
#####################
# inferences vs real values
result=pd.DataFrame()
result["date"]=df["date"]
result["real_searches"]=df["unemployment"]
result["inferred_results"]=pd.DataFrame(rfecv.predict(X))  

result["inferred_results"]=result["inferred_results"].apply(lambda x: 0 if x<0 else round(x,2))
result.to_csv("../tmp/results-inferences-overwrite.csv")
#result.to_csv('gs://your bucket in GCS/results-inferences-overwrite.csv') 

print("csv of inferences to tmp folder done")

# weekly score

score = pd.DataFrame({"date": [today], 'RMSE': [round(rfecv.score(X_train, target_train),4)]})
score.to_csv("../tmp/results-weekly_rmse-append.csv")
# score.to_csv('gs://your bucket in GCS/results-weekly_rmse-append.csv')

# Ranking of how important are the following keywords to infer in Google searches in Spain
# the keyword "unemployment"
features=pd.DataFrame()
features["features"]=X.columns
features["top_important"]=rfecv.ranking_
features.sort_values(by=["top_important"], inplace=True)
features.reset_index(drop=True, inplace=True)
features.to_csv("../tmp/results-ranking_of_features-overwrite.csv")
# features.to_csv('gs://your bucket in GCS/results-ranking_of_features-overwrite.csv')

print("csv of scores and evolution of features to tmp folder done")