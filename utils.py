import numpy as np
import torch as tr
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


def discretize(x,num_bins):
    if np.unique(x).size <= num_bins: return x
    lb,ub=min(x),max(x)
    bins=np.linspace(lb,ub,num_bins+1)
    x_disc=np.digitize(x,bins)
    return x_disc

def MLE_loss(y_dist, y, w):
    return -tr.mean(y_dist.log_prob(y)*w)

# def pinballLossDist(x_dist, q, x, w):
#     q=tr.FloatTensor(q)
#     n_points,n_quantiles=x.shape[0],Q.shape[0]
#     z=x_dist.icdf(Q)
#     diff=x-z
#     for r in range(diff.shape[0]):
#         for c in range(diff.shape[1]):
#             if diff[r,c] >= 0: 
#                 diff[r,c] *= Q[c]
#             else:
#                 diff[r,c] *= -(1-Q[c])
#     diff*=w # accounting for the observation weights
#     return tr.mean(diff)

# def pinballLoss_v0(Q, q, X, w):
#     assert Q.shape[0]==X.shape[0]
#     assert Q.shape[1]==len(q)
#     n_points=X.shape[0]
#     diff=X-Q
#     for i in range(diff.shape[0]):
#         for j in range(diff.shape[1]):
#             if diff[i,j] >= 0: 
#                 diff[i,j] *= q[j]
#             else:
#                 diff[i,j] *= -(1-q[j])
#     diff*=w # accounting for the observation weights
#     return tr.mean(diff)

def pinballLoss(Q, q, X, w):
    q=tr.FloatTensor(q)
    assert Q.shape[0]==X.shape[0]
    assert Q.shape[1]==len(q)
    n_points=X.shape[0]
    diff=(X-Q)*w # weighted residual
    return tr.mean( tr.max(q*diff,(q-1)*diff))

def getBatch(df, row_ids, covar_names, target_name):
    x_batch = df.loc[row_ids][covar_names].to_numpy()
    y_batch = df.loc[row_ids][target_name].to_numpy().reshape(-1,1)
    w_batch = df.loc[row_ids]['wi'].to_numpy().reshape(-1,1)
    return x_batch, y_batch, w_batch

# def getBatch(df, X_sparse, row_ids, categorical_vars, numeric_vars, target_var,scaled_weights=True):
#     x_batch = sc.sparse.hstack([X_sparse[var][row_ids,:] for var in categorical_vars]).todense()
#     x_batch = np.concatenate((x_batch,df.iloc[row_ids][numeric_vars].to_numpy()),axis=1)
#     x_batch = x_batch
#     y_batch = df.iloc[row_ids][target_var].to_numpy().reshape(-1,1)
#     if scaled_weights and 'wts' in df:
#         w_batch = df.iloc[row_ids]['wts'].to_numpy().reshape(-1,1)
#     else:
#         w_batch = df.iloc[row_ids]['wi'].to_numpy().reshape(-1,1)
#     return x_batch, y_batch, w_batch



def preProcess(df):
       
    # Reassigning the education category "0" equal to "9" (so as to have caterories [9,10,11,12,13,14,15,16,17]) 
    df['education']=df['education'].apply(lambda x: 9 if x==0 else x)
    
    # Removing the rows with negative income or expenses
    df = df.loc[df['income']>=0]
    if 'expense' in df: df = df.loc[df['expense']>=0] 
    # Adding a column with scaled weights
    if 'wi' in df:
        total=sum(df['wi'])
        df['wts']=df['wi']/total
        
    #Adding a small constant (1.0) to the income/expense values that are zero and adding column with log-income /log-expense
    df['income']=df['income'].apply(lambda x: 1. if x==0 else x)
    df['log_income']=np.log(df['income'])
    if 'expense' in df: 
        df['expense']=df['expense'].apply(lambda x: 1. if x==0 else x)
        df['log_expense']=np.log(df['expense'])
        
    # replacing Nans with -1
    df=df.replace(np.nan,-1)

    return df


# def preProcess(df):
#     # replacing Nans with -1
#     df=df.replace(np.nan,-1)
    
#     # Making the education category 0 equal to 9 (so as to have caterories [9,10,11,12,13,14,15,16,17] 
#     df['education']=df['education'].apply(lambda x: 9 if x==0 else x)
    
#     # Removing the rows with negative income or expenses
#     df = df.loc[df['income']>=0]
#     if 'expense' in df: df = df.loc[df['expense']>=0] 
#     # Adding a column with scaled weights
#     if 'wi' in df:
#         total=sum(df['wi'])
#         df['wts']=df['wi']/total
        
#     #Adding a small constant (1.0) to the income/expense values that are zero and adding column with log-income /log-expense
#     df['income']=df['income'].apply(lambda x: 1. if x==0 else x)
#     df['log_income']=np.log(df['income'])
#     if 'expense' in df: 
#         df['expense']=df['expense'].apply(lambda x: 1. if x==0 else x)
#         df['log_expense']=np.log(df['expense'])

#     # splitting the data into two parts (when the expense is zero and otherwise)
#     if 'expense' in df:
#         df_zero_expense = df.loc[df['expense']<2]
#         df_nonzero_expense = df.loc[df['expense']>2]
#         return df_nonzero_expense, df_zero_expense
#     else:
#         return df


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


# def hotEncodeData(df,categorical_vars):
#     # hot encoding the categorical variables
#     hot_encoded_data={}
#     num_categories = {}
#     for var in categorical_vars:
#         x = df[var]
#         unique_cats=np.sort(pd.unique(x))
#         enc=OneHotEncoder(drop='first',categories=[unique_cats],dtype=np.int32)
#         enc.fit(x.to_numpy().reshape(-1,1))
#         hot_encoded_data[var] = enc.transform(x.to_numpy().reshape(-1,1))
#         num_categories[var]=len(unique_cats)-1 
#     return hot_encoded_data, num_categories
    
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
    
    