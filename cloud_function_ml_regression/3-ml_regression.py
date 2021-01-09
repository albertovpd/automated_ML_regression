import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA as pca
from scipy import stats

from datetime import datetime

from my_functions import variance_threshold_selector

from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor



df=pd.read_csv("../tmp/dataset_final_processed.csv",index_col=[0])
low_v = df.drop(columns=["date","unemployment","outliers_score" ])  
X_raw= variance_threshold_selector(low_v, 5) # removing values than vary less than 5%

removed=[]
for c in low_v.columns:
    if c not in X_raw.columns:
        #print(c)
        removed.append(c)
removed=pd.DataFrame(removed, columns =["Removed_columns"])
removed.to_csv("../tmp/removed_features_low_variance.csv")
#removed.to_csv("gs://--yourbucket--/removed_features_low_variance.csv")   #<========================== 3

#normalizer = preprocessing.MinMaxScaler()
#X = pd.DataFrame(normalizer.fit_transform(X_raw))
standardizer = preprocessing.StandardScaler()
X = pd.DataFrame(standardizer.fit_transform(X_raw))

X.columns= X_raw.columns
X.shape
target=df["unemployment"]

# all but last 4 rows
X_train=X.iloc[:-4]
target_train=target.iloc[:-4]
# last 4 rows, which are the ones I have no data and I want to infer
X_test=X.iloc[-4:]
target_test=target.iloc[-4:]

# for csvs
today=datetime.now().date()

#----------------- ML -----------------
# Minimal number of features to play with
min_number_features =  df.shape[0]//10
regression = Lasso(alpha=0.1)
#regression = LinearRegression()
rfecv = RFECV(estimator=regression,
              step=2, 
              min_features_to_select=min_number_features, 
              cv=KFold(n_splits=10,
                    shuffle=True,
                    random_state=42),
              scoring='neg_mean_squared_error')
rfecv.fit(X_train, target_train)

#weekly_score
weekly_score= pd.DataFrame()
weekly_score["rmse"]=pd.Series(rfecv.score(X_train, target_train))
weekly_score["date"]=pd.to_datetime(today)
weekly_score.to_cvs("../tmp/weekly_score.csv")
#weekly_score.to_csv("gs://--yourbucket--/weekly_score.csv") #<========================= 4

# comparison real searches vs inferred results
inferred_results=list(rfecv.predict(X)) # this is just to avoid generating nans in df
unemployment=list(df.unemployment)
result=pd.DataFrame()
result["date"]=df["date"]
result["real_searches"]=unemployment
result["inferred_results"]=inferred_results
result["inferred_results"]=result["inferred_results"].apply(lambda x: 0 if x<0 else round(x,2))
if result["inferred_results"].min()==0:
    result.drop(result.tail(1).index,inplace=True) # drop last row (better visualization)
result.to_csv("../tmp/results-inferences-overwrite.csv")
#result.to_csv("gs://--yourbucket--/results-inferences-overwrite.csv") # <=============================== 5


number_features=pd.DataFrame()
number_features["number_columns"]=pd.Series(rfecv.n_features_)
number_features["date"]= pd.to_datetime(today)
number_features.to_csv("../tmp/number_features_this_week.csv")
#number_features.to_csv("gs://--yourbucket--/number_features_this_week.csv") # <===================== 6

# features performance
plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=34, pad=20)
plt.xlabel('Number of features selected', fontsize=28, labelpad=20)
plt.ylabel('Inner algorithm score', fontsize=28, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
plt.savefig("../tmp/RFE_columns.png")   # <================================== 7

# Ranking of how important are the following keywords to infer in Google searches in Spain
# the keyword "unemployment"
ranking_features=pd.DataFrame()
ranking_features["features"]=X_train.columns
ranking_features["top_important"]=rfecv.ranking_
ranking_features.sort_values(by="top_important", ascending=True, inplace=True, ignore_index=True)
ranking_features.to_csv("../tmp/ranking_of_features.csv")
#ranking_features.to_csv("gs://ml_regression/ranking_of_features.csv")  # <============================ 8

print("ML regression done")