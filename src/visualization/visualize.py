import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import create_category_interactions
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler



def plot_categorical_target_dist(df, categorical_features_names,
                                 target_name, plot_slice, figsize, plt_adjust_left=0.4, log_counts=False):
    """
    Plot target distribution for a categorical variable.
    
    PARAMETERS:
            df: Pandas DataFrame which contains the features to groupby and the target.
            categorical_features_name_list: List categroical featrues' names (strings) to groupby 
            target_name: List with name of the target.
            plot_slice: Tuple input for the slice to plot. ex: top 25 would be (0,25)
            
           
    Author: KMP    
    """
    n_features = len(categorical_features_names)
    
    data = df.loc[:, categorical_features_names + target_name ]
    data.fillna("NA",inplace=True)
    data = data.groupby( categorical_features_names )[target_name[0]].agg(['count','mean'])
    data.reset_index(inplace=True)
    data = data.sort_values(['mean','count'], ascending=False)
    
    if (log_counts):
        data["count"] = data["count"].apply(lambda x: np.log(1 + x))
    
    print(data.columns)
    
    if (n_features > 1):
        groupby_feature = '-'.join(categorical_features_names)
        data[groupby_feature] = create_category_interactions(data, categorical_features_names)
    else:                                    
        groupby_feature = categorical_features_names[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey='col', figsize=figsize)
    
    sub = slice(plot_slice[0],plot_slice[1])

    sns.barplot(x="mean", y= groupby_feature, 
                data=data.iloc[sub,:], orient='h', ax=ax1)

    sns.barplot(x="count", y= groupby_feature, 
                data=data.iloc[sub,:], orient='h', ax=ax2)
    

        
    
    ax2.get_yaxis().set_visible(False)
    
    fig.subplots_adjust(left=plt_adjust_left)
    
    if (log_counts):
        plt.xlabel("log counts")
        fig.suptitle('Target Mean and Category Log Counts', x=0.8 , fontsize=16, ha = 'right')
    else:
        fig.suptitle('Target Mean and Category Counts', x=0.8 , fontsize=16, ha = 'right')
    
    plt.show()
    
    return



def correlation_plot(df, axislabel, title, n_clusters=10, figsize=(10,10), cmap='viridis'):
    """
    plot correlation of all variables supplied in df and group by KMeans
    
    PARAMETERS:
        df : [Pandas DataFrame]
        axislabel : str
            Label for x,y axis of correlation plot
        title : str, 
            Plot title 
        n_clusters = int, optional, default: 10
            Number of clusters in KMeans
        figsize= tuple, default: (10,10)
        
    RETURNS:
        feture_clusters: tuple, length 2
            (cluster_labels, sorted_feature_index)
        
    Author: KMP
    """
    
    # correlation
    corr = df.corr()
    
    # fit kmeans
    km = KMeans(n_clusters=n_clusters, random_state=123)
    km.fit(corr.fillna(0).to_sparse())
    cluster_labels = km.labels_
    sorted_feature_index = np.argsort(km.labels_)
    
    # reshuffle according to 
    new_corr = corr.iloc[np.argsort(km.labels_),np.argsort(km.labels_)]
    
    # plot
    fig = plt.figure(figsize=(10,10))
    ax = plt.gca()    
    im = ax.matshow(new_corr, cmap = cmap)
    fig.colorbar(im)
    
    # adjust ticklabels
    tick_labels, tick_locations = np.unique(cluster_labels[sorted_feature_index], return_index=True)
    xticks = ax.set_xticks(tick_locations)
    yticks = ax.set_yticks(tick_locations)
    xticklabels = ax.set_xticklabels(tick_labels)
    yticklabels = ax.set_yticklabels(tick_labels)
    
    ax.set_xlabel(axislabel, fontdict={"fontsize":12})
    ax.set_ylabel(axislabel, fontdict={"fontsize":12})
    ax.set_title(title, fontdict={"fontsize":16})
    plt.show()
    plt.close('all')

    return cluster_labels, sorted_feature_index


def plot_feature_importance(estimator, dftrain):
    "feature importance plot"
    
    minmax = MinMaxScaler()

    features = dftrain.columns
    importances = minmax.fit_transform(clf.feature_importances_.reshape(-1,1)).reshape(-1)
    indices = np.argsort(importances)[-30:]

    plt.figure(figsize=(12,10))
    plt.title('Feature Importance - Top 30', fontdict={"fontsize":16})
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance (scaled to [0,1])', fontdict={"fontsize":14})

    plt.show()
