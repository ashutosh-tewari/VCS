import numpy as np
import pandas as pd
import math as m
from sklearn.metrics import mutual_info_score
import utils as utl
import scipy as sc
from collections import OrderedDict
# importing NN library
import torch as tr
import torch.nn as nn
# importing torch distributions and transformations 
import torch.distributions as trd
import torch.distributions.transforms as trt

# Conditional Gaussian Model via DNN parameterization
# DNN parameterizes the mean and the log-sig functions
class ConditionalModel(nn.Module):
    def __init__(self, input_size, layers_size):
        super(ConditionalModel, self).__init__()
        
        # mean model
        layers=OrderedDict()
        size_in=input_size
        for i in range(len(layers_size)):
            size_out = layers_size[i]
            layers[f'mu_linear{i}']=nn.Linear(size_in,size_out)
            layers[f'mu_batchnorm{i}']=nn.BatchNorm1d(size_out)
            layers[f'mu_relu{i}']=nn.ReLU()
            size_in=size_out
        layers['mu']=nn.Linear(size_in,1)
        self.mu= nn.Sequential(layers)
        
        # log_sig model
        layers=OrderedDict()
        size_in=input_size
        for i in range(len(layers_size)):
            size_out = layers_size[i]
            layers[f'sig_linear{i}']=nn.Linear(size_in,size_out)
            layers[f'sig_batchnorm{i}']=nn.BatchNorm1d(size_out)
            layers[f'sig_relu{i}']=nn.ReLU()
            size_in=size_out
        layers['logSig']=nn.Linear(size_in,1)
        self.log_sig= nn.Sequential(layers)
    
    def forward(self,x):
        return (self.mu(x),self.log_sig(x))
    
    
# Quantile Regression Model via DNN parameterization
# DNN parameterizes the quantiles 
class QuantileRegressionModel(nn.Module):
    def __init__(self, input_size, layers_size, num_quantiles):
        super(QuantileRegressionModel, self).__init__()
        layers=OrderedDict()
        size_in=input_size
        for i in range(len(layers_size)):
            size_out = layers_size[i]
            layers[f'linear{i}']=nn.Linear(size_in,size_out)
            layers[f'batchnorm{i}']=nn.BatchNorm1d(size_out)
            layers[f'relu{i}']=nn.ReLU()
            size_in=size_out
        layers['quantiles']=nn.Linear(size_in,num_quantiles)
        self.model= nn.Sequential(layers)
    
    def forward(self,x):
        return self.model(x)
 