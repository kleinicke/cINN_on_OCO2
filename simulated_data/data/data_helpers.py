import numpy as np
import torch
try:
    import data.dataloader as dataloader
except:
    import dataloader as dataloader
########################################
#            Load dataset              #
########################################
batch_size = 512
test_ana_loader, train_ana_loader = dataloader.get_ana_loaders(batch_size)

def concatenate_all_set(cut=None):
    x_all, y_all, ana_all = [], [], []

    for idx,(x,y,ana) in enumerate(test_ana_loader):
        x_all.append(x)
        y_all.append(y)
        ana_all.append(ana)
        if cut:
            if idx*batch_size>cut:
                break
    for idx, (x,y,ana) in enumerate(train_ana_loader):
        x_all.append(x)
        y_all.append(y)
        ana_all.append(ana)
        if cut:
            if idx*batch_size>cut:
                break
    x_cat, y_cat, ana_all = torch.cat(x_all, 0), torch.cat(y_all, 0), torch.cat(ana_all, 0)

    if cut:
        x_cat, y_cat, ana_all = x_cat[:cut], y_cat[:cut], ana_all[:cut]
    return x_cat, y_cat, ana_all

def concatenate_train_set(cut=None):
    x_all, y_all, ana_all = [], [], []

    for idx,(x,y,ana) in enumerate(train_ana_loader):
        x_all.append(x)
        y_all.append(y)
        ana_all.append(ana)
        if cut:
            if idx*batch_size>cut:
                break

    x_cat, y_cat, ana_all = torch.cat(x_all, 0), torch.cat(y_all, 0), torch.cat(ana_all, 0)

    if cut:
        x_cat, y_cat, ana_all = x_cat[:cut], y_cat[:cut], ana_all[:cut]
    return x_cat, y_cat, ana_all

def concatenate_test_set(cut=None):
    x_all, y_all, ana_all = [], [], []

    for idx,(x,y,ana) in enumerate(test_ana_loader):
        x_all.append(x)
        y_all.append(y)
        ana_all.append(ana)
        if cut:
            if idx*batch_size>cut:
                break
    x_cat, y_cat, ana_all = torch.cat(x_all, 0), torch.cat(y_all, 0), torch.cat(ana_all, 0)

    if cut:
        x_cat, y_cat, ana_all = x_cat[:cut], y_cat[:cut], ana_all[:cut].numpy()
    return x_cat, y_cat, ana_all


def concatenate_set(data_set, cut=None, batch_size = batch_size):
    x_all, y_all, ana_all = [], [], []

    for idx,(x,y,ana) in enumerate(data_set):
        x_all.append(x)
        y_all.append(y)
        ana_all.append(ana)
        if cut:
            if idx*batch_size>cut:
                break
    x_cat, y_cat, ana_all = torch.cat(x_all, 0), torch.cat(y_all, 0), torch.cat(ana_all, 0)

    if cut:
        x_cat, y_cat, ana_all = x_cat[:cut], y_cat[:cut], ana_all[:cut].numpy()
    return x_cat, y_cat, ana_all


