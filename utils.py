import numpy as np
import torch as tr
import pandas as pd
import scipy as sc
from sklearn.preprocessing import OneHotEncoder


def discretize(x,num_bins):
    if np.unique(x).size <= num_bins: return x
    lb,ub=min(x),max(x)
    bins=np.linspace(lb,ub,num_bins+1)
    x_disc=np.digitize(x,bins)
    return x_disc

def MLE_loss(x_dist, x, w):
    return -tr.mean(x_dist.log_prob(x)*w),''

def pinballLoss(x_dist, x, w, q=tr.FloatTensor([0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995])):
    n_points,n_quantiles=x.shape[0],q.shape[0]
    z=x_dist.icdf(q)
    diff=x-z
    for r in range(diff.shape[0]):
        for c in range(diff.shape[1]):
            if diff[r,c] >= 0: 
                diff[r,c] *= q[c]
            else:
                diff[r,c] *= -(1-q[c])
    diff*=w # accounting for the observation weights
    return tr.mean(diff),  z

def pinballLoss_know_qunatiles(quantiles, x, w):
    q,z=list(quantiles.keys()),list(quantiles.values())
    n_points,n_quantiles=x.shape[0],len(q)
    diff=x-z
    for r in range(diff.shape[0]):
        for c in range(diff.shape[1]):
            if diff[r,c] >= 0: 
                diff[r,c] *= q[c]
            else:
                diff[r,c] *= -(1-q[c])
    diff*=w # accounting for the observation weights
    return tr.mean(diff),  z

def getBatch(df, X_sparse, row_ids, categorical_vars, numeric_vars, target_var,scaled_weights=True):
    x_batch = sc.sparse.hstack([X_sparse[var][row_ids,:] for var in categorical_vars]).todense()
    x_batch = np.concatenate((x_batch,df.iloc[row_ids][numeric_vars].to_numpy()),axis=1)
    x_batch = x_batch
    y_batch = df.iloc[row_ids][target_var].to_numpy().reshape(-1,1)
    if scaled_weights and 'wts' in df:
        w_batch = df.iloc[row_ids]['wts'].to_numpy().reshape(-1,1)
    else:
        w_batch = df.iloc[row_ids]['wi'].to_numpy().reshape(-1,1)
    return x_batch, y_batch, w_batch


def preProcess(df):
    # Removing the rows with negative income or expenses
    df = df.loc[df['income']>=0]
    if 'expense' in df: df = df.loc[df['expense']>=0] 
    # Adding a column with scaled weights
    if 'wi' in df:
        total=sum(df['wi'])
        df['wts']=df['wi']/total
    #Adding a small constant to the income and expense values that are zero
    df['income']=df['income'].apply(lambda x: abs(np.random.randn())*1E-3 if x==0 else x)
    if 'expense' in df: df['expense'] = df['expense'].apply(lambda x: abs(np.random.randn())*1E-3 if x==0 else x)
    # Adding columns with log-income and log-expense values
    df['log_income']=np.log(df['income'])
    if 'expense' in df: df['log_expense']=np.log(df['expense'])
    # Replacing nan values with -1
    df=df.replace(np.nan,-1) 
    return df


def hotEncodeData(df,categorical_vars):
    # hot encoding the categorical variables
    hot_encoded_data={}
    num_categories = {}
    for var in categorical_vars:
        x = df[var]
        unique_cats=np.sort(pd.unique(x))
        enc=OneHotEncoder(drop='first',categories=[unique_cats],dtype=np.int32)
        enc.fit(x.to_numpy().reshape(-1,1))
        hot_encoded_data[var] = enc.transform(x.to_numpy().reshape(-1,1))
        num_categories[var]=len(unique_cats)-1 
    return hot_encoded_data, num_categories
    
    
    
    