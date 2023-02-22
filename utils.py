import numpy as np
import torch as tr
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import math as m
import pandas as pd


# Discretization of 1-d numpy array
def discretize(x,num_bins):
    if np.unique(x).size <= num_bins: return x
    lb,ub=min(x),max(x)
    bins=np.linspace(lb,ub,num_bins+1)
    x_disc=np.digitize(x,bins)
    return x_disc

# Negative log-likelihood given a distribution (from torch.distributions class) and observations
def MLE_loss(y_dist, y, w=1):
    return -tr.mean(y_dist.log_prob(y)*w)

# Pinball loss. 
# X: n X 1 observations, Q: n X |q| quantiles, q: list of prob. levels, w: n X  1 weights
def pinballLoss(Q, q, X, w):
    q=tr.FloatTensor(q)
    assert Q.shape[0]==X.shape[0]
    assert Q.shape[1]==len(q)
    n_points=X.shape[0]
    diff=(X-Q)*w # weighted residual
    return tr.mean( tr.max(q*diff,(q-1)*diff))

# Function to get a batch of points from a pandas table (used for DNN learning)
def getBatch(df, row_ids, covar_names, target_name):
    x_batch = df.loc[row_ids][covar_names].to_numpy()
    y_batch = df.loc[row_ids][target_name].to_numpy().reshape(-1,1) if target_name else []
    w_batch = df.loc[row_ids]['wi'].to_numpy().reshape(-1,1) if 'wi' in df else []
    return x_batch, y_batch, w_batch


# Preprocessing the consumer expense data
def preProcess(df, split=0.75):
    # Removing the rows with negative income or expenses
    df = df.loc[df['income']>=0]
    if 'expense' in df: df = df.loc[df['expense']>0] 
    # Replacing NaNs with -1
    df=df.replace(np.nan,-1)
    # Reassigning the education category "0" equal to "9" (so as to have caterories [9,10,11,12,13,14,15,16,17]) 
    df['education']=df['education'].apply(lambda x: 9 if x==0 else x)
    # Further splittling the table (for training and validation)
    if split:
        # first, shuffling training dataset before splitting (sample with fraction =1 shuffles all the rows)
        np.random.seed(0)
        df=df.sample(frac=1)
        n_trn = int(len(df)*split)
        # then, splitting
        df_2=df.iloc[n_trn:]
        df_1=df.iloc[:n_trn]
        return df_1,df_2
    else:
        return df

    
# Hot Encode categorical variables in a pandas table and return another table with hot-encoded data
def hotEncodeData(df,categorical_vars):
    # hot encoding the categorical variables
    df_hot_encoded=pd.DataFrame(index=df.index)
    column_names = []
    for var in categorical_vars:
        x = df[var]
        unique_cats=np.sort(pd.unique(x))
        n_cats = len(unique_cats)
        enc=OneHotEncoder(drop='first',categories=[unique_cats],dtype=np.int32,sparse=False)
        enc.fit(x.to_numpy().reshape(-1,1))
        hot_encoded_array = enc.transform(x.to_numpy().reshape(-1,1))
        names=[f'{var}{i}' for i in range(n_cats-1)]
        temp_df = pd.DataFrame(hot_encoded_array,index=df.index, columns=names)
        df_hot_encoded = pd.concat([df_hot_encoded,temp_df], ignore_index=False, axis=1)
        column_names += names 
    return df_hot_encoded, column_names


# Function for Scatter plotting True vs. Pred (with p90 prediction interval)    
def plotQuantiles(tru, quantiles, num_samples=50, axis_labels={'x':'True value', 'y':'Estimated Value'}, log_scale=True):
    ids=np.random.choice(tru.shape[0],num_samples, replace=False)
    tru=tru[ids]
    quantiles=quantiles[ids,:]
    p05,p50,p95=quantiles[:,0], quantiles[:,1], quantiles[:,2]
    lb=(p50-p05).reshape(1,-1)
    ub=(p95-p50).reshape(1,-1)
    errors=np.concatenate((lb,ub),axis=0)
    if log_scale:
        plt.loglog(tru, p50, 'rs')
    else:
         plt.plot(tru, p50, 'rs')
    plt.errorbar(tru,p50,yerr=errors,ls='none')
    plt.xlabel(axis_labels['x'],fontsize=14)
    plt.ylabel(axis_labels['y'],fontsize=14)
    plt.title('True vs Estimated (with 95% CI)')
    
    