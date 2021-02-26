import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.font_manager
from mpl_toolkits.mplot3d import Axes3D

from sklearn.feature_selection import VarianceThreshold
from sklearn import decomposition 

#from google.cloud import storage


def plot_dimensions(df_outliers,df_date,dates):
    '''
    This is a terrible function, don't kill me. Fast and furious mode.
    '''
    #2D
    pca=decomposition.PCA()
    pca.n_components=2
    pca_data=pca.fit_transform(df_outliers)
    pca_data=pd.DataFrame(pca_data)
    pca_data.rename(columns={0:"a",1:"b"}, inplace=True)

    pca_data["date"]=df_date
    pca_data["outliers"]=[0 if d in list(dates) else 1 for d in list(pca_data.date)]

    red= pca_data[pca_data["outliers"]==1][["a","b"]]
    blue= pca_data[pca_data["outliers"]==0][["a","b"]]

    fig = plt.figure(figsize=[15,15])
    ax = fig.add_subplot(111)
    ax.scatter(red.a, red.b, marker="v", color="r") # "P"
    ax.scatter( blue.a,blue.b, marker="^", color="b") # "P"
    ax.grid(True)
    plt.xlabel('Dimension a')
    plt.ylabel('Dimension b')

    plt.title('PCA of 125 columns to dashboard detected outliers')
    plt.savefig("../tmp/outliers_2d.png")

    #3D, the very same, don't kill me
    pca=decomposition.PCA()
    pca.n_components=3
    pca_data=pca.fit_transform(df_outliers)
    pca_data=pd.DataFrame(pca_data)
    pca_data.rename(columns={0:"a",1:"b",2:"c"}, inplace=True)

    pca_data["date"]=df_date
    pca_data["outliers"]=[0 if d in list(dates) else 1 for d in list(pca_data.date)]

    red= pca_data[pca_data["outliers"]==1][["a","b","c"]]
    blue= pca_data[pca_data["outliers"]==0][["a","b","c"]]

    fig = plt.figure(figsize=[15,15])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(red.c,red.a, red.b, marker="v", color="r") # "P"
    ax.scatter( blue.c,blue.a,blue.b, marker="^", color="b") # "P"
    ax.grid(True)
    plt.xlabel('Dimension a')
    plt.ylabel('Dimension b')

    plt.title('PCA of 125 columns to dashboard detected outliers')
    plt.savefig("../tmp/outliers_3d.png")
    

def scientific_rounding(value):
    error=str(value[1]).split(".")
    if error[0]!="0":
        
        return str(int(value[0]))+"+-"+str(int(value[1]))
    else:
        d=0
        for e in list(error[1]):
            if e=="0":
                d+=1
            else:
                break
        return str(round(value[0],d+2))+"+-"+str(round(value[1],d+2))

        
def variance_threshold_selector(data, threshold):
    '''
    It removes any attribute (column) than vary less than the percentaje of the threshold
    '''
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]
    

def outliers_graph(df, outlier_method, outliers_begin, threshold, xmin, xmax):
    
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(xmin, xmax, 100))
    Z = outlier_method.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(20, 14))
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, xmax), cmap=plt.cm.Blues_r)
    #a = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')
    #plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')
    b = plt.scatter(df.iloc[:outliers_begin, 0], df.iloc[:outliers_begin, 1], c='white', s=20, edgecolor='k')
    c = plt.scatter(df.iloc[outliers_begin:, 0], df.iloc[outliers_begin:, 1], c='black', s=20, edgecolor='k')
    plt.axis('tight')
    '''
    # In case I'll use the axis unemployment and 1D proyection
    plt.ylim((-5,105)) # forcing the graph to fit my target range
    plt.title('Elliptic envelope over my target and a 1D proyection of my training set', fontsize=20)
    plt.xlabel('1D proyection', fontsize=16)
    plt.ylabel('Unemployment', fontsize=16)
    #
    '''
    plt.legend(
        [b,c],["Valid instances", "outliers"],
        #[a.collections[0], b, c],
        #['Decision boundary', 'Valid instances', 'Outliers'],
        prop=matplotlib.font_manager.FontProperties(size=34),
        loc='lower right')
    plt.savefig('../tmp/outliers.png') #<========================================
    #plt.show()   


def creating_dataset(df,column):
    '''
    Column is the column in which are allocated the keywords, for every case: political, social and economical columns
    '''
    df["Date"]=pd.to_datetime(df["Date"])
    # list of new columns
    list_keywords=list(set(df[column]))
    # creating empty dataframe to append info
    df_final=pd.DataFrame()
    df_final["date"]=list(set(df["Date"]))
    
    for k in list_keywords:
        # creating a new dataframe for every keyword in the column, getting the occurrences of keyword and mean of sentiment
        df4=pd.DataFrame()
        df4=df[df[column]==k].groupby(["Date"]).agg(['count','mean'])
        # erase multiindex
        df4.columns=df4.columns.droplevel(0)
        # this will be our score, occurrences * mean 
        df4[k]=df4["count"]*df4["mean"]
        # date column to perform the join by it
        df4["date"]=df4.index
        df4.drop(columns=["count","mean"],inplace=True)
        # this is where we combine the empty dataset, every keyword in its place
        df_final=df_final.merge(df4,how='left', left_on='date', right_on='date')

    # resampling 
    # this is weird: transform date column in index, group by, then transform again index in column, to make the further join
    df_final.index=df_final["date"]
    df_final = df_final.resample('W-SUN').mean() #weekly totals
    df_final.sort_values(by="date", ascending=True, inplace=True)
    df_final["date"]=df_final.index
    df_final.reset_index(drop=True, inplace=True)    
    
    return df_final

