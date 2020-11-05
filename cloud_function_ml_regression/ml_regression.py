import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
#from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

# There's room for improvement in this script, and I love it. Stuff for myself from the future.

df=pd.read_csv("../tmp/dataset_final_processed.csv")
df.drop(columns='Unnamed: 0', inplace=True)

# Transform into standard normal distribution using the z-score definition
X = df.drop(columns=["date","unemployment"]) #returns a numpy array
X = X.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
target=df["unemployment"]

# all but last 4 rows
X_train=X.iloc[:-4]
target_train=target.iloc[:-4]

# last 4 rows, which are the ones I have no data and I want to infer
X_test=X.iloc[-4:]
target_test=target.iloc[-4:]

# alpha=1 is like a regular regression. for getter performance use elastic net instead (l1&l2 mix).
regression = Lasso(alpha=0.1)
#regression = LinearRegression() # oher model

# neg mean squared error. it always is negative but what you get is the positive representation
rfecv = RFECV(estimator=regression, step=1, min_features_to_select=15, cv=5,scoring='neg_mean_squared_error')
rfecv.fit(X_train, target_train)

# results for bigquery
result=pd.DataFrame()
result["date"]=df["date"]
result["real_searches"]=df["unemployment"]
result["infered_results"]=pd.DataFrame(rfecv.predict(X))

result["infered_results"]=result["infered_results"].apply(lambda x: 0 if x<0 else round(x,2))
result.to_csv("../tmp/results.csv")

# weekly score
score = pd.DataFrame({"date": [max(df["date"])], 'RMSE': [round(rfecv.score(X_train, target_train),4)]})
score.to_csv("../tmp/weekly_score.csv")

# Ranking of how important are the following keywords to infer in Google searches in Spain
# the keyword "unemployment"

features=pd.DataFrame()
features["features"]=X.columns
features["top_important"]=rfecv.ranking_
features.sort_values(by=["top_important"], inplace=True)
features.reset_index(drop=True, inplace=True)
features.to_csv("../tmp/ranking_of_features.csv")