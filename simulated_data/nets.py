#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import copy

import config as c
import data.dataloader as dataloader

import data.prepare_data as prepare_data

import time

#from nets_older import *

#import visual.nn_nn as nn


flatten = lambda l: [item for sublist in l for item in sublist]

class Cnn_long_solve(nn.Module):
    def __init__(self):
        super().__init__()
        #do_rate1=0.3
        do_rate=c.fn_dropout#0.3

        self.cnn = nn.Sequential(
                    nn.Conv1d(in_channels=1,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.BatchNorm1d(20),
                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    #nn.ReLU(True),
                    #nn.BatchNorm1d(20),
                    #nn.ReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(True),
                    #nn.MaxPool1d(kernel_size=2, stride=2, padding=0,return_indices=False),
                    nn.BatchNorm1d(20),
                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    #nn.BatchNorm1d(20),
                    #nn.ReLU(True),
                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    #nn.ReLU(True),
                    #nn.BatchNorm1d(20),
                    #nn.ReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(True),
                    #nn.MaxPool1d(kernel_size=2, stride=2, padding=0,return_indices=False),
                    nn.BatchNorm1d(20),
                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    #nn.ReLU(True),
                    #nn.BatchNorm1d(20),
                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    #nn.ReLU(True),
                    #nn.BatchNorm1d(20),
                    #nn.ReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.BatchNorm1d(20),
                    #nn.ReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(True),
                    #nn.MaxPool1d(kernel_size=2, stride=2, padding=0,return_indices=False),
                    nn.BatchNorm1d(20),
                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    #nn.ReLU(True),
                    #nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(True),
                    #nn.MaxPool1d(kernel_size=2, stride=2, padding=0,return_indices=False),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(True),
                    nn.BatchNorm1d(20),
                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    #nn.BatchNorm1d(20),
                    #nn.ReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(True),
                    #nn.MaxPool1d(kernel_size=2, stride=2, padding=0,return_indices=False),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    #nn.ReLU(True),
                    #nn.BatchNorm1d(20),
                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    #nn.ReLU(True),
                    #nn.BatchNorm1d(20),
                    #nn.Dropout(p=do_rate2),
                    #nn.ReLU(True),
                    #nn.Conv1d(in_channels=20,out_channels=10, kernel_size=3, stride=1, padding=1),
                    #nn.ReLU(True),
                    )

        self.smaller = nn.Sequential(

                    nn.Linear(780+c.params_in_spectrum, 1024),
                    #nn.Dropout(p=do_rate),
                    #nn.Linear(1024, 1024),
                    nn.ReLU(True),
                    nn.Linear(1024, 256),
                    nn.Dropout(p=do_rate),

        )                    
        self.linear = nn.Sequential(
                    #nn.BatchNorm1d(256),

                    #nn.Dropout(p=do_rate1),
                    nn.Linear(256, 2048), #c.y_dim_in is 1497
                    #nn.BatchNorm1d(2048),
                    nn.ReLU(True),
                    nn.Dropout(p=do_rate),
                    nn.Linear(2048, 512),
                    nn.ReLU(True),
                    #nn.BatchNorm1d(512),
                    nn.Linear(512, 1024),
                    #nn.BatchNorm1d(1024),
                    nn.ReLU(True),
                    nn.Dropout(p=do_rate),
                    nn.Linear(1024, 1024),
                    #nn.BatchNorm1d(1024),
                    nn.ReLU(True),
                    nn.Dropout(p=do_rate),
                    nn.Linear(1024, 1024),
                    nn.ReLU(True),
                    nn.Dropout(p=do_rate),
                    #nn.Linear(128, 1024),
                    #nn.ReLU(False),
                    #nn.Dropout(p=do_rate2),
                    #nn.Linear(1024, 512),
                    #nn.ReLU(True),
                    #nn.Dropout(p=0),
                    #nn.BatchNorm1d(1024),
                    nn.Linear(1024, 167),
                    )

        self.fc_final = nn.Linear(167, c.x_dim)

        self.linear1 = torch.nn.Linear(c.params_in_spectrum, 780 + c.params_in_spectrum)

    def forward_solve(self, x):
        x2=x[:,-c.params_in_spectrum:]
        x=x[:,None,:-c.params_in_spectrum]
        x = self.cnn(x).view(x.size(0),-1)
        x = torch.cat((x, x2), 1)
        x = self.smaller(x)
        #print("jo")
        #x = self.linear1(x2)
        x = self.linear(x)
        return self.fc_final(x)

    def features(self, x):
        x2=x[:,-c.params_in_spectrum:]
        x=x[:,None,:-c.params_in_spectrum]
        x = self.cnn(x).view(x.size(0),-1)
        x = torch.cat((x, x2), 1)
        x = self.smaller(x)
        #x = self.linear(x)
        #print("jo")
        #x = self.linear1(x2)
        #x = self.smaller(x)
        return x
        #return x
class Cnn_long_solve3(nn.Module):
    def __init__(self):
        super().__init__()
        #do_rate1=0.3
        do_rate=c.fn_dropout#0.3

        self.cnn = nn.Sequential(
                    nn.Conv1d(in_channels=1,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(True),
                    nn.BatchNorm1d(20),

                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(True),
                    nn.BatchNorm1d(20),

                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(True),
                    nn.BatchNorm1d(20),

                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(True),
                    nn.BatchNorm1d(20),

                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),

                    )

        self.smaller = nn.Sequential(

                    nn.Linear(780+c.params_in_spectrum, 1024),
                    nn.Dropout(p=do_rate),
                    #nn.Linear(1024, 1024),
                    #nn.ReLU(True)
                    nn.Linear(1024, 512),
                    nn.Dropout(p=do_rate),

        )                    



    def features(self, x):
        x2=x[:,-c.params_in_spectrum:]
        x=x[:,None,:-c.params_in_spectrum]
        #print(x.shape)
        x = self.cnn(x).view(x.size(0),-1)
        #print(x.shape)
        x = torch.cat((x, x2), 1)
        #x = self.linear(x)
        #print(x.shape)
        #assert 0, x.shape
        #assert 0
        return self.smaller(x)
        #return x


class Cnn_sep_solve(nn.Module):
    def __init__(self):
        super().__init__()
        #do_rate1=0.3
        do_rate=c.fn_dropout#0.3

        self.cnn = nn.Sequential(
                    nn.Conv1d(in_channels=3,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    nn.Conv1d(in_channels=60,out_channels=60, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(60),
                    )

        self.cnn1 = nn.Sequential(
                    nn.Conv1d(in_channels=1,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(20),
                    nn.LeakyReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.LeakyReLU(),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(20),
                    nn.LeakyReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.LeakyReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(20),
                    nn.LeakyReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.LeakyReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.LeakyReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(20),
                    nn.LeakyReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.LeakyReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(20),
                    nn.LeakyReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(20),
                    nn.LeakyReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(20),
                    nn.LeakyReLU(True),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm1d(20),
                    )
            
        self.cnn2 = copy.deepcopy(self.cnn1)
        self.cnn3 = copy.deepcopy(self.cnn1)

        self.smaller = nn.Sequential(

                    nn.Linear(1920+c.params_in_spectrum, 1024), #for seperate spectra
                    #nn.Linear(1298, 1024), #for *3 spectra
                    nn.LeakyReLU(True),
                    nn.Dropout(p=do_rate),
                    #nn.Linear(1024, 1024),
                    #nn.ReLU(True)
                    nn.Linear(1024, 256),
                    nn.Dropout(p=do_rate),

        )                    
        self.linear = nn.Sequential(
                    #nn.BatchNorm1d(256),

                    #nn.Dropout(p=do_rate1),
                    nn.Linear(256, 2048), #c.y_dim_in is 1497
                    #nn.BatchNorm1d(2048),
                    nn.Dropout(p=do_rate),
                    nn.Linear(2048, 512),
                    #nn.ReLU(True),
                    #nn.BatchNorm1d(512),
                    nn.Linear(512, 1024),
                    #nn.BatchNorm1d(1024),
                    nn.LeakyReLU(True),
                    nn.Dropout(p=do_rate),
                    nn.Linear(1024, 1024),
                    #nn.BatchNorm1d(1024),
                    #nn.ReLU(True),
                    nn.Dropout(p=do_rate),
                    nn.Linear(1024, 1024),
                    #nn.ReLU(True),
                    nn.Dropout(p=do_rate),
                    #nn.Linear(128, 1024),
                    #nn.ReLU(False),
                    #nn.Dropout(p=do_rate2),
                    #nn.Linear(1024, 512),
                    #nn.ReLU(True),
                    #nn.Dropout(p=0),
                    #nn.BatchNorm1d(1024),
                    nn.Linear(1024, 167),
                    )

        self.fc_final = nn.Linear(167, c.x_dim)
        self.fc_final2 = copy.deepcopy(self.fc_final)

        self.spec_solver = nn.Sequential(

            nn.Linear(1920, 2048),
            nn.Dropout(p=do_rate),
            nn.Linear(2048, 512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(True),
            nn.Dropout(p=do_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(p=do_rate),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(p=do_rate),
            nn.Linear(1024, 167),
            )

        self.param_solver = nn.Sequential(
            nn.Linear(c.params_in_spectrum, 1024), #780 # todo split spectra
            nn.Dropout(p=do_rate),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(True),
            nn.Linear(1024, 256),
            nn.LeakyReLU(True),
            nn.Dropout(p=do_rate),
            nn.Linear(256, 2048), 
            nn.LeakyReLU(True),
            nn.Dropout(p=do_rate),
            nn.Linear(2048, 512),
            nn.LeakyReLU(True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(True),
            nn.Dropout(p=do_rate),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(True),

            nn.Dropout(p=do_rate),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(True),
            nn.Dropout(p=do_rate),

            nn.Linear(1024, 167),
            )

        self.gradients = None
        self.gradients2 = None

        self.last_small = nn.Sequential(
            nn.Linear(167*2, 167*2),
            nn.LeakyReLU(),
            nn.Linear(167*2, 167),
            nn.LeakyReLU(True)
            )

    def forward_solve(self, x):
        x2=x[:,-c.params_in_spectrum:]
        #print(x[0,:5],x[0,c.spec_length:c.spec_length+5],x[0,2*c.spec_length:2*c.spec_length+5])
        #print(x[0,:10])
        x = x[:,:-c.params_in_spectrum].view(x.size(0),-1,c.spec_length)
        #print(x[0,0,:10])
        #print(x[0,1,:10])
        #print(x[0,2,:10])
        #print(x[0,0,:5],x[0,1,:5],x[0,2,:5])
        #x=x[:,None,:-c.params_in_spectrum]
        #print(x.shape)
        #x1_1 = x[:,:,:c.spec_length]
        #x1_2 = x[:,:,c.spec_length:2*c.spec_length]
        #x1_3 = x[:,:,2*c.spec_length:3*c.spec_length]
        #print(x1_1.shape)
        """
        x1_1 = self.cnn1(x[:,0,:][:,None,:]).view(x.size(0),-1)
        x1_2 = self.cnn2(x[:,1,:][:,None,:]).view(x.size(0),-1)
        x1_3 = self.cnn3(x[:,2,:][:,None,:]).view(x.size(0),-1)
        x = torch.cat((x1_1, x1_2), 1)
        x = torch.cat((x, x1_3), 1)
        x = x.view(x.size(0),-1)
        """
        x = self.cnn(x).view(x.size(0),-1)
        x = torch.cat((x, x2), 1)
        x = self.smaller(x)
        x = self.linear(x)
        return self.fc_final(x)

    def spec_solve(self, x):
        #x2=x[:,-c.params_in_spectrum:]
        x=x[:,None,:-c.params_in_spectrum]
        #print(x.shape)
        x1_1 = x[:,:,:c.spec_length]
        x1_2 = x[:,:,c.spec_length:2*c.spec_length]
        x1_3 = x[:,:,2*c.spec_length:3*c.spec_length]
        #print(x1_1.shape)
        x1_1 = self.cnn1(x1_1).view(x.size(0),-1)
        x1_2 = self.cnn2(x1_2).view(x.size(0),-1)
        x1_3 = self.cnn3(x1_3).view(x.size(0),-1)
        x = torch.cat((x1_1, x1_2), 1)
        x = torch.cat((x, x1_3), 1)
        #x = torch.cat((x, x2), 1)
        #x = self.smaller(x)
        x = self.spec_solver(x)
        return x#self.fc_final(x)

    def param_solve(self, x):
        x2=x[:,-c.params_in_spectrum:]
        #print(x2.shape)
        #x = self.smaller(x)
        x = self.param_solver(x2)
        return x#self.fc_final(x)

    def nothing_solve(self, x):
        x2 = x[:,-c.params_in_spectrum:]-100
        m= nn.ReLU()
        x2=m(x2)
        #print(x2)
        #print(x2.shape)
        #print(x2.shape)
        #x = self.smaller(x)
        x = self.param_solver(x2)
        return x

    def old_features(self, x):
        #x.register_hook(self.activations_hook)
        x2=x[:,-c.params_in_spectrum:]
        x=x[:,None,:-c.params_in_spectrum]
        x = self.cnn(x).view(x.size(0),-1)
        x = torch.cat((x, x2), 1)
        x.register_hook(self.activations_hook)
        #x = self.linear(x)
        return self.smaller(x)
        #return x

    #def last_small(self,x):

    #    return last_small(x)

    def fn_func(self, x):
        x = self.forward_solve(x)
        return x
        """
        x.requires_grad = True
        x.register_hook(self.activations_hook)

        x1 = self.param_solve(x)
        #x = self.nothing_solve(x)
        x2 = self.spec_solve(x)

        x = torch.cat((x1, x2), 1)
        #print(x1.shape,x2.shape,x.shape)
        x = self.last_small(x)
        
        return self.fc_final(x)
        """

    def features_seperate(self,x):
        x2=x[:,-c.params_in_spectrum:]
        #print(x[0,:5],x[0,c.spec_length:c.spec_length+5],x[0,2*c.spec_length:2*c.spec_length+5])
        #print(x[0,:10])
        x = x[:,:-c.params_in_spectrum].view(x.size(0),-1,c.spec_length)

        x = self.cnn(x).view(x.size(0),-1)
        x = torch.cat((x, x2), 1)
        x = self.smaller(x)

        #x1 = self.param_solve(x)
        ###x = self.nothing_solve(x)
        #x2 = self.spec_solve(x)

        return x#torch.cat((x1, x2), 1)

    def features(self, x):
        return self.features_seperate(x)

    def activations_hook(self, grad):
        self.gradients = grad

    def activations_hook2(self, grad):
        self.gradients2 = grad

    def get_activations_gradient(self):
        return self.gradients

    def get_activations_gradient2(self):
        return self.gradients2

class Cnn_long_sep(nn.Module):
    def __init__(self):
        super().__init__()
        #do_rate1=0.3
        do_rate=c.fn_dropout#0.3

        lista = [nn.LeakyReLU(True), nn.BatchNorm1d(20), 
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1)]
        flatten = lambda l: [item for sublist in l for item in sublist]
        self.cnn = nn.Sequential(
                    nn.Conv1d(in_channels=1,out_channels=20, kernel_size=3, stride=1, padding=1),
                    *flatten([lista for i in range(3)]),
                    #nn.LeakyReLU(True),
                    #nn.BatchNorm1d(20),
                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    #nn.LeakyReLU(True),
                    #nn.BatchNorm1d(20),

                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    #nn.LeakyReLU(True),
                    #nn.BatchNorm1d(20),

                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
                    *flatten([lista for i in range(2)]),
                    #nn.LeakyReLU(True),
                    #nn.BatchNorm1d(20),
                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    #nn.LeakyReLU(True),
                    #nn.BatchNorm1d(20),

                    #nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20)

                    )

        self.smaller = nn.Sequential(

                    nn.Linear(1920+c.params_in_spectrum, 512), #+c.params_in_spectrum 780 # todo split spectra
                    nn.Dropout(p=do_rate),
                    #nn.Linear(1024, 1024),
                    #nn.ReLU(True)
                    #nn.Linear(1024, 512),
                    #nn.Dropout(p=do_rate),

        )     
        self.smaller1 = nn.Sequential(

                    nn.Linear(1920, 1024), #+c.params_in_spectrum 780 # todo split spectra
                    nn.Dropout(p=do_rate),
                    #nn.Linear(1024, 1024),
                    #nn.ReLU(True)
                    nn.Linear(1024, 512-128),
                    nn.Dropout(p=do_rate),

        )  
        self.smaller2 = nn.Sequential(

                    nn.Linear(c.params_in_spectrum, 128), #+c.params_in_spectrum 780 # todo split spectra
                    nn.Dropout(p=do_rate),
                    #nn.Linear(1024, 1024),
                    #nn.ReLU(True)
                    nn.Linear(128, 128),
                    nn.Dropout(p=do_rate),

        )  
        self.last_small = nn.Sequential(
            nn.Linear(512, 512), #+c.params_in_spectrum 780 # todo split spectra
            nn.Dropout(p=do_rate),
            #nn.Linear(1024, 1024),
            nn.LeakyReLU(True),
            nn.Linear(512, c.x_dim)
        )




    def features(self, x):
        x2=x[:,-c.params_in_spectrum:]
        x=x[:,None,:-c.params_in_spectrum]#.view(x.size(0),3,-1)
        x = self.cnn(x).view(x.size(0),-1)
        x = torch.cat((x, x2), 1)

        return self.smaller(x)

    def fn_func(self, x):
        x = self.features(x)
        x = self.last_small(x)
        return x

class Cnn_long_sep2(nn.Module):
    def __init__(self):
        super().__init__()
        #do_rate1=0.3
        do_rate=c.fn_dropout#0.3

        self.cnn = nn.Sequential(
                    nn.Conv1d(in_channels=1,out_channels=20, kernel_size=5, stride=2, padding=2),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=5, stride=2, padding=2),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=5, stride=2, padding=2),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=5, stride=2, padding=2),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),
                    nn.Conv1d(in_channels=20,out_channels=20, kernel_size=5, stride=2, padding=2),
                    nn.LeakyReLU(True),
                    nn.BatchNorm1d(20),

                    )

        self.smaller = nn.Sequential(

                    nn.Linear(1920+c.params_in_spectrum, 1024),
                    nn.Dropout(p=do_rate),
                    #nn.Linear(1024, 1024),
                    #nn.LeakyReLU(True),
                    nn.Linear(1024, 512),
                    nn.Dropout(p=do_rate),

        )     


    def features(self, x):
        #x.register_hook(self.activations_hook)
        x2=x[:,-c.params_in_spectrum:]
        x=x[:,None,:-c.params_in_spectrum]#.view(x.size(0),3,-1)
        x = self.cnn(x).view(x.size(0),-1)
        x = torch.cat((x, x2), 1)
        #x = x2
        #x.register_hook(self.activations_hook)
        #x = self.linear(x)
        return self.smaller(x)
        #return x



class seperate_hard_solve(nn.Module):
    def __init__(self):
        super().__init__()
        #do_rate1=0.3
        do_rate=c.fn_dropout#0.3

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=3,out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(20),
            nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(20),
            nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(20),
            nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(20),
            nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(20),
            nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(20),
            nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(20),
            nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(20),
            nn.Conv1d(in_channels=20,out_channels=20, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(20),

        )   
        self.spec_solver = nn.Sequential(

            nn.Linear(2540, 2048),
            nn.LeakyReLU(True),
            #nn.Dropout(p=do_rate),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(1024),
            #nn.Dropout(p=do_rate),
            #nn.Linear(1024, 1024),
            #nn.ReLU(True),
            #nn.Dropout(p=do_rate),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(1024),
            #nn.Dropout(p=do_rate),
            nn.Linear(1024, 256),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(256),
            )

        self.param_solver = nn.Sequential(
            nn.Linear(c.params_in_spectrum, 1024), #780 # todo split spectra
            nn.LeakyReLU(True),
            nn.BatchNorm1d(1024),
            #nn.Dropout(p=do_rate),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(1024),

            #nn.Dropout(p=do_rate),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(True),

            #nn.Dropout(p=do_rate),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(True),

            nn.BatchNorm1d(1024),
            #nn.Dropout(p=do_rate),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(1024),
            #nn.Dropout(p=do_rate),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(1024),
            #nn.Dropout(p=do_rate),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(1024),
            #nn.Dropout(p=do_rate),

            #nn.Dropout(p=do_rate),
            nn.Linear(1024, 256),
            nn.LeakyReLU(True),
            )
        """
            nn.Linear(1024, 256),
            nn.LeakyReLU(True),
            nn.Dropout(p=do_rate),
            nn.Linear(256, 2048), 
            nn.LeakyReLU(True),
            nn.Dropout(p=do_rate),
            nn.Linear(2048, 512),
            nn.LeakyReLU(True),

            nn.Linear(512, 1024),
            nn.LeakyReLU(True),
        """
        self.combi = nn.Sequential(
            nn.Linear(256*2, 256*2),
            nn.LeakyReLU(),
            nn.Dropout(c.fn_dropout),
            #nn.BatchNorm1d(512),
            #nn.Dropout(p=do_rate),
            nn.Linear(256*2, 256*2),
            nn.LeakyReLU(),
            nn.Dropout(c.fn_dropout),
            #nn.BatchNorm1d(512),
            #nn.Dropout(p=do_rate),
            nn.Linear(256*2, 256*2),
            nn.LeakyReLU(),
            nn.Dropout(c.fn_dropout),
            #nn.BatchNorm1d(512)
            )

        self.last_small = nn.Sequential(
            nn.Linear(256*2, 256),
            nn.LeakyReLU(True),
            #nn.BatchNorm1d(256),
            nn.Dropout(c.fn_dropout),
            nn.Linear(256, c.x_dim)
            )


    def base(self,x):
        x1 = self.param_solver(x[:,-c.params_in_spectrum:])
        #x = self.nothing_solve(x)
        x2 = x[:,:-c.params_in_spectrum].view(x.size(0),-1,c.spec_length)
        x2 = self.cnn(x2).view(x2.size(0),-1)
        x2 = self.spec_solver(x2)

        x = torch.cat((x1, x2), 1)
        #print(x1.shape,x2.shape,x.shape)
        return self.combi(x)

    def fn_func(self, x):
        x = self.base(x)
        x = self.last_small(x)
        return x

    def features(self, x):
        x = self.base(x)
        end = nn.Sequential(*list(self.last_small.children())[:-1])
        return x#end(x)

    def forward_solve(self,x):
        return self.fn_func(x)

    def forward(self,x):
        return self.forward_solve(x)

    def relprop(self, r):
        for l in range(len(self.last_small), 0, -1):
            r = self.last_small[l - 1].relprop(r)

        print("last small over",r.shape)
        r1 = r[:,:256]
        r2 = r[:,256:]
        r_class = r
        for l in range(len(self.param_solver), 0, -1):
            print("r1",r1.shape)
            r1 = self.param_solver[l - 1].relprop(r1)
            print(r1.sum(),r1[0,:10])

        print("r1_final",r1.sum(),r1)
        for l in range(len(self.spec_solver), 0, -1):
            r2 = self.spec_solver[l - 1].relprop(r2)
        
        r2 = r2.view(r2.size(0),20,-1)
        print(r2.shape)
        for l in range(len(self.cnn), 0, -1):
            print("r2",r2.shape)
            r2 = self.cnn[l - 1].relprop(r2)
            print(r2.sum(),r2[0,0,:10])

        r2 = r2.view(r2.size(0),-1)
        
        print("r1,r2 shape",r1.shape,r2.shape)
        r = torch.cat((r1, r2), 1)
        print("and combined",r.shape)

        return r, r_class




name=c.feature_net_name

if name == "fully_connected":
    model = FullyConnected().to(c.device)
elif name == "cnn_long_solve":
    model = Cnn_long_solve().to(c.device)#Cnn_long_solve().to(c.device)
elif name == "cnn_sep_solve":
    model = Cnn_sep_solve().to(c.device)#Cnn_long_solve().to(c.device)
elif name == "Easy_solve":
    model = Easy_solve().to(c.device)#Cnn_long_solve().to(c.device)
elif name == "Harder_solve":
    model = Harder_solve().to(c.device)
elif name == "Cnn_add_sep_solve":
    model = Cnn_add_sep_solve().to(c.device)
elif name == "seperate_hard_solve":
    model = seperate_hard_solve().to(c.device)
elif name == "Cnn_long_combined_solve":
    model = Cnn_long_combined_solve().to(c.device)
elif name == "FullyConnectedBig":
    model = FullyConnectedBig().to(c.device)
elif name == "Cnn_long_sep":
    model = Cnn_long_sep().to(c.device)
elif name == "Cnn_short_solve":
    model = Cnn_short_solve().to(c.device)
elif name == "PriorNet":
    model = PriorNet().to(c.device)

    
    
else:
    assert 0, "Wrong featurenet Name"

def read_params():
    params=[]
    names=[]
    for name,param in model.named_parameters():

        params.append(torch.sum(param.data).detach().cpu().numpy())
        names.append(name)
    return params, names