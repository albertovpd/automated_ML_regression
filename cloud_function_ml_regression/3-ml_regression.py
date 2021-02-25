import pandas as pd
import numpy as np
import collections

from sklearn import preprocessing
from sklearn.decomposition import PCA as pca
from scipy import stats

from datetime import datetime

from my_functions import variance_threshold_selector

import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold, cross_validate #cross_val_score
#
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import make_scorer
import sklearn.metrics as metrics 
from math import sqrt
import collections

df=pd.read_csv("../tmp/dataset_final_processed.csv",index_col=[0])

#--  removing values than vary less than 5%
low_v = df.drop(columns=["date","unemployment" ])  
X_raw= variance_threshold_selector(low_v, 5) 

removed=[]
for c in low_v.columns:
    if c not in X_raw.columns:
        #print(c)
        removed.append(c)
removed=pd.DataFrame(removed, columns =["Removed_columns"])
    removed.to_csv("../tmp/removed_features_low_variance.csv")

#-- normalization/standarization
normalizer = preprocessing.MinMaxScaler()
X = pd.DataFrame(normalizer.fit_transform(X_raw))
#standardizer = preprocessing.StandardScaler()
#X = pd.DataFrame(standardizer.fit_transform(X_raw))

X.columns= X_raw.columns
X.shape
target=df["unemployment"]
# all but last 4 rows
X_train=X.iloc[:-4]
target_train=target.iloc[:-4]
# last 4 rows, which are the ones I have no data and I want to infer
X_test=X.iloc[-4:]
target_test=target.iloc[-4:]

# for results
today=datetime.now().date()

#----------------- ML metrics -----------------
regression={"Linear reg":   LinearRegression(),
            "Lasso":    Lasso(alpha=0.1, selection="random", max_iter=10000, random_state=42),
            "Rigde":    Ridge(alpha=0.1,max_iter=10000, solver='auto', random_state=42)
            }

metrics_dict= {
    #"r2": metrics.r2_score, # already checked
    "mae": make_scorer(metrics.mean_absolute_error,
            greater_is_better=False),
    "rmse": make_scorer(lambda y,y_pred: sqrt(metrics.mean_squared_error(y,y_pred)),
            greater_is_better=False),
    "mape": make_scorer(lambda y,y_pred: np.mean(np.abs(y-y_pred)/y)*100,
            greater_is_better=False),
    "mean_squared_error": make_scorer(metrics.mean_squared_error,
            greater_is_better=False),
    "explained_variance": make_scorer(metrics.explained_variance_score,
            greater_is_better=False),
    "max_error": make_scorer(metrics.max_error,
            greater_is_better=False),
    "median_absolute_error": make_scorer(metrics.median_absolute_error,
            greater_is_better=False)
}

#----------------- ML -----------------
min_number_features =  df.shape[0]//8 # Min number of features to play with

# dictionary of r2 metrics: we'll use this for choosing the model
model_performance={}
for r in regression:
    rfecv = RFECV(estimator=regression[r],
                step=1, 
                  min_features_to_select=min_number_features, 
                  cv=KFold(n_splits=10,
                        shuffle=True,
                        random_state=42),
                  scoring='r2')
    rfecv.fit(X_train, target_train)
    score = rfecv.score(X_train, target_train)
    
    if score <=0.99:
        model_performance[score]=[r,regression[r]]
    
# this is to choose the winner model        
model_performance = collections.OrderedDict(sorted(model_performance.items(), reverse=True))
score=list(model_performance.keys())[0]
model_name=list(model_performance.values())[0][0]
model_config=list(model_performance.values())[0][1]

print(score, model_name)

# final dataframe
results={"date":today,"model":model_name, "r2":round(score,3)}

# Ranking of how feature importance
ranking_features=pd.DataFrame()
ranking_features["features"]=X_train.columns
ranking_features["top_important"]=rfecv.ranking_
ranking_features.sort_values(by="top_important", ascending=True, inplace=True, ignore_index=True)
ranking_features.to_csv("../tmp/ranking_of_features.csv")  # <============================

# now I have the chosen features, let's use cross validation with that columns:
chosen_features=list(ranking_features[ranking_features["top_important"]<=10]["features"])
X_train= X[chosen_features].iloc[:-4]
X_test=X[chosen_features].iloc[-4:]
len(X_train.columns)

# results
results["selected_columns"]=len(X_train.columns)

# Validation of model with the dictionary of metrics. I'll get a measure for every validation folder.
evaluation = cross_validate(model_config, X_train, target_train,
                cv = KFold(n_splits=10), scoring = metrics_dict)

# just in case
for e in evaluation:
    evaluation[e] = evaluation[e][~np.isnan(evaluation[e])]
    evaluation[e] = evaluation[e][~np.isinf(evaluation[e])]

# populating our df of scores
results["mae"]=round(evaluation["test_mae"].mean(),3)
results["mae_error"]=round(evaluation["test_mae"].std(),3)

results["rmse"]=round(evaluation["test_rmse"].mean(),3)
results["rmse_error"]=round(evaluation["test_rmse"].std(),3)

results["mape"]=round(evaluation["test_mape"].mean(),3)
results["mape_error"]=round(evaluation["test_mape"].std(),3)

results["mse"]=round(evaluation["test_mean_squared_error"].mean(),3)
results["mse_error"]=round(evaluation["test_mean_squared_error"].std(),3)

results["explained_var"]=round(evaluation["test_explained_variance"].mean(),3)
results["explained_var_error"]=round(evaluation["test_explained_variance"].std(),3)

results["median_abs_error"]=round(evaluation["test_median_absolute_error"].mean(),3)
results["median_abs_error_eror"]=round(evaluation["test_median_absolute_error"].std(),3)

results_df=pd.DataFrame.from_dict(results,orient='index').T
results_df.to_csv("../tmp/weekly_score.csv") #<============================================


# inferences plot
inferred_results=list(rfecv.predict(X)) # this is just to avoid generating nans in df
unemployment=list(df.unemployment)
date=df.date
result=pd.DataFrame()
result["date"]=df["date"]
result["real_searches"]=unemployment
result["inferred_results"]=inferred_results
result["inferred_results"]=result["inferred_results"].apply(lambda x: 0 if x<0 else round(x,2))
if result["inferred_results"].min()==0:
    result.drop(result.tail(1).index,inplace=True) # drop last row (better visualization)
result.to_csv("../tmp/results-inferences-overwrite.csv") # <=============================== 5

# features performance plot
plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=34, pad=20)
plt.xlabel('Number of features selected', fontsize=28, labelpad=20)
plt.ylabel('Inner algorithm score', fontsize=28, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
plt.savefig("../tmp/RFE_columns.png")   # <================================== 7

#
print("ML regression done")