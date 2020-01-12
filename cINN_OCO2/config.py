"""
Sets all the parameter for training and evaluation. 
For the datasettings see data/data_config.py
"""
import torch
import os
import numpy as np
import data.prepare_data as prepare_data
#In dataconfig some other important configurations are done
import data.data_config as dc
import re
import __main__


main_file = __main__.__file__.split("/")[-1][:-3]
run_name = "train"

params_in_spectrum = prepare_data.params_in_spectrum #5
spec_length = prepare_data.spec_length
print(f"number of external parameter is {params_in_spectrum}")
co2_pos = 0
param_names = prepare_data.params_names
ana_names = prepare_data.ana_names
element_names = list(prepare_data.namedict.keys())
arbitrary_number = 1
print(f"number of resulting parameter is {len(param_names)}")
######################
#  General settings  #
######################

load_file = f'output/co2INN{arbitrary_number}.pt'+"1"
load_file = ''

file_it = 1
if load_file:
    file_it = re.findall(r'\d+', load_file)[-1]
    file_it = int(file_it)+1

filename = f'output/co2INN{arbitrary_number}.pt{file_it}'


print("File name is",filename)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device is {device}")

assert device == 'cuda', "When cuda isn't active for some reason, but should work, restart it with the commands in the lines below"
"""sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
"""


live_visualization = True
progress_bar = True

checkpoint_save_interval = 30#5#64#120 
checkpoint_save_overwrite = False
#######################
#  Training schedule  #
#######################

lr_init = 1e-4
decay_by = 0.005
lr_reduce_factor = 4*5e-2
#dependent on the memory of the GPU
batch_size = 1024

n_epochs = 11
n_its_per_epoch = 2**10
do_rev = False


l2_weight_reg = 1e-5
adam_betas = (0.9, 0.999)

pre_low_lr =1 
latent_noise = 1.

blockTestData = True
stop_train_loss=1e10

#####################
#  Data dimensions  #
#####################

import data.dataloader as dataloader

x_dim = (dataloader.parameter_size) 
y_dim_in = dataloader.spectrum_size 
y_dim_features = 128

test_ana_loader, train_ana_loader = dataloader.get_ana_loaders(batch_size)
year_sets = dataloader.year_sets

###########
#  Model  #
###########

n_blocks = 10#10#25#50#18#18#12
internal_width = int(256)#*2)#512
use_permutation = True
exponent_clamping = 1.0
init_scale = 0.075/2
fc_dropout = 0.
fn_dropout = 0.2 #for Featurenet
skip_large = False

#####################
#  FeatureNet Setup #
#####################

feature_net_name = "cnn_long_solve"

print("Uses fn_model:",feature_net_name)

################
#  Evaluation  #
################

evaluation_samples = 1028
N_post = 512
losses = []
sigma = []
mu = []
params_mask = prepare_data.params_mask
spec_delete = prepare_data.spec_delete

###########
#  Names  #
###########


configuration = f"{feature_net_name}_{main_file[:5]}"

import model
model = model.model
model.to(device)
