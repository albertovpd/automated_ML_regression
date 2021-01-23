import pandas as pd
import numpy as np
import collections

from sklearn import preprocessing
from sklearn.decomposition import PCA as pca
from scipy import stats

from datetime import datetime

from my_functions import variance_threshold_selector, scientific_rounding

from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import matplotlib.font_manager

from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.ensemble import RandomForestRegressor
#
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


from sklearn.metrics import mean_squared_error, explained_variance_score, max_error, mean_absolute_error,median_absolute_error, r2_score

df=pd.read_csv("../tmp/dataset_final_processed.csv",index_col=[0])
low_v = df.drop(columns=["date","unemployment" ])  
X_raw= variance_threshold_selector(low_v, 5) # removing values than vary less than 5%

removed=[]
for c in low_v.columns:
    if c not in X_raw.columns:
        #print(c)
        removed.append(c)
removed=pd.DataFrame(removed, columns =["Removed_columns"])
removed.to_csv("../tmp/removed_features_low_variance.csv")
#removed.to_csv("gs://--yourbucket--/removed_features_low_variance.csv")   #<========================== 3

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

# for csvs
today=datetime.now().date()

#----------------- ML -----------------
regression={"Linear":   LinearRegression(),
            "Lasso":    Lasso(alpha=0.1, selection="random", max_iter=10000, random_state=42),
            "Rigde":    Ridge(alpha=0.1,max_iter=10000, solver='auto', random_state=42)
            }

min_number_features =  df.shape[0]//10 # Min number of features to play with

model_performance={}
for m in regression:
    rfecv = RFECV(estimator=regression[m],
                step=1, 
                  min_features_to_select=min_number_features, 
                  cv=KFold(n_splits=10,
                        shuffle=True,
                        random_state=42),
                  scoring='neg_mean_squared_error')
    rfecv.fit(X_train, target_train)
    score = rfecv.score(X_train, target_train)
    if score <=0.99:
        model_performance[score]=[m,regression[m]]

model_performance = collections.OrderedDict(sorted(model_performance.items(), reverse=True)) #sort by score
# this is according how I built the model_performance_dictionary
score=list(model_performance.keys())[0] 
model_name=list(model_performance.values())[0][0]
model_config=list(model_performance.values())[0][1]

print("winner model: ", model_name, ",score: ",score)
#print(model_performance)
#regression= LinearRegression()
# regression = Lasso(alpha=0.1,
#                   selection="random",
#                   max_iter=10000,
#                   random_state=42)
#regression= Ridge(alpha=0.1,max_iter=10000, solver='auto', random_state=42)

# overfitting ñapa turbomierder XXL omg x3
# random_number=42
# for _ in list(range(10)):                       
#     rfecv = RFECV(estimator=regression,
#                 step=1, 
#                   min_features_to_select=min_number_features, 
#                   cv=KFold(n_splits=10,
#                         shuffle=True,
#                         random_state=random_number),
#                   scoring='neg_mean_squared_error')
                       
#     rfecv.fit(X_train, target_train)
#     score = rfecv.score(X_train, target_train)
#     random_number+=1
#     if score<0.99:
#         break
            
# print("3 - Score: ",score)

# Ranking of how important are the following keywords to infer in Google searches in Spain
# the keyword "unemployment"
ranking_features=pd.DataFrame()
ranking_features["features"]=X_train.columns
ranking_features["top_important"]=rfecv.ranking_
ranking_features.sort_values(by="top_important", ascending=True, inplace=True, ignore_index=True)
ranking_features.to_csv("../tmp/ranking_of_features.csv")
#ranking_features.to_csv("gs:///ranking_of_features.csv")  # <============================ 8


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

#-------------------- weekly_scores with standard deviation ------------------------------------
mean = result.real_searches.head(-4).mean()
N = len(result.real_searches.head(-4))
y_real=result.real_searches.head(-4) 
y_pred=result.inferred_results.head(-4)
mape=round(np.abs( ((y_real-y_pred)/y_real).sum() /N )*100,3)

metrics={"date":[today], 
         "selected_columns":[rfecv.n_features_],
         "model":model_name,
        "mape":[mape],
         "alg_rmse":[round(score,3)]}

metrics_cross_val_score=[
                        "neg_root_mean_squared_error",
                        "neg_mean_squared_error",
                         "r2",
                         "explained_variance",
                         "neg_mean_absolute_error",
                         "max_error",
                         "neg_median_absolute_error"
                        ]

regression=model_config # the winner model before
X_train= X_train[ranking_features[ranking_features["top_important"]==1]["features"]] #selecting the same columns than before
for m in metrics_cross_val_score:
    score=cross_val_score(regression, 
        X_train,
        target_train, 
        cv=KFold(n_splits=10, shuffle=True, random_state=42), scoring=m)

    score= [-score.mean()/mean,score.std()/mean]    

    metrics[m]=round(score[0],2)
    dev="std_"+m
    metrics[dev]=round(score[1],2)
    if m=="neg_root_mean_squared_error":
        metrics["rmse"]=round(score[0],2)
        dev="std_rmse"
        metrics[dev]=round(score[1],2)
        metrics["std_up"]=[round(score[0]+score[1],3)]
        metrics["std_down"]=[round(score[0]-score[1],3)]

weekly_score=pd.DataFrame(metrics)
weekly_score.rename(columns={
                   'neg_root_mean_squared_error': "rmse", 
                   'std_neg_root_mean_squared_error': "error_rmse",
                   'neg_mean_squared_error':"mse", 
                   'std_neg_mean_squared_error':"error_mse", 
                   'std_r2':"error_r2",
                   'std_explained_variance':"error_explained_variance",
                    'neg_mean_absolute_error':"mae", 
                   'std_neg_mean_absolute_error':"error_mae", 
                   'std_max_error':"error_max_error", 
                   'neg_median_absolute_error':"median_ae",
                   'std_neg_median_absolute_error':"error_median_ae"}, inplace=True)

weekly_score.to_csv("../tmp/weekly_score.csv") #<=====================
#weekly_score.to_csv("gs://--yourbucket--/weekly_score.csv") #<========================= 4


# features performance plot
plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=34, pad=20)
plt.xlabel('Number of features selected', fontsize=28, labelpad=20)
plt.ylabel('Inner algorithm score', fontsize=28, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
plt.savefig("../tmp/RFE_columns.png")   # <================================== 7


print("ML regression done")