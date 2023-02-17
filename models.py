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

class ConditionalExpenseModel(nn.Module):
    def __init__(self, input_size, layers_size):
        super(ConditionalExpenseModel, self).__init__()
        layers=OrderedDict()
        size_in=input_size
        for i in range(len(layers_size)):
            size_out = layers_size[i]
            layers[f'linear{i}']=nn.Linear(size_in,size_out)
            layers[f'rely{i}']=nn.ReLU()
            size_in=size_out
        layers['mu_logSig']=nn.Linear(size_in,2)
        self.model= nn.Sequential(layers)
    
    def forward(self,x):
        out=self.model(x)
        mu,log_sig=out[:,0],out[:,1]
        return (mu,log_sig)