"""
Sets all the parameter for training and evaluation. 
For the datasettings see data/data_config.py
"""
import torch
import os
import numpy as np
import data.prepare_data as prepare_data
import re
#print("import config \n\n\n\n\n")


#main_file = __main__.__file__.split("/")[-1][:-3]
run_name = "train"

#params_in_spectrum = prepare_data.params_in_spectrum #5
params_in_spectrum = prepare_data.number_external_params #5

#spec_length = prepare_data.spec_length
print(f"number of external parameter is {params_in_spectrum}")
co2_pos = len(prepare_data.output_params[0])-4
skip_inputs = prepare_data.skip_inputs
element_names = prepare_data.element_names[skip_inputs:]
#co2_pos = 0
param_names = element_names #prepare_data.params_names
#param_names = param_names[-4:]
#ana_names = prepare_data.ana_names
#element_names = list(prepare_data.namedict.keys())

#if len(param_names)>9:
#    arbitrary_number = 1
#elif len(param_names)>6:
#    arbitrary_number = 2
#else:
arbitrary_number = 1
#print(f"number of resulting parameter is {len(param_names)}")
######################
#  General settings  #
######################

#load_file = 'output/co2INN1_18.pt_checkpoint_0480_2'
load_file = f'output/co2INN{arbitrary_number}.pt'+"1"
load_file = ''

file_it = 1
if load_file:
    file_it = re.findall(r'\d+', load_file)[-1]
    file_it = int(file_it)+1

#load_checkpoit = 0#30
filename = f'output/co2INN{arbitrary_number}.pt{file_it}'

number_of_trainingdata_sets = prepare_data.number_of_additional_trainingdata_sets + 1
if number_of_trainingdata_sets > 1:
    add1 = prepare_data.additional_trainingdata_sets[0]
data = f"{number_of_trainingdata_sets}_{add1}_"
filename = f'output/co2INN{data}{params_in_spectrum}.pt'
#if load_checkpoit:
#    filename += f"_checkpoint_{load_checkpoit:04d}"
#filename = 'output/co2INN1_28.pt_checkpoint_0064'

print("File name is",filename)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device is {device}")

assert device == 'cuda'
#When cuda isn't active anymore for some reason restart it with the commands:
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

lr_init = 1e-3#*0.1#*1e-1#3e-4#*1e-1 #1e-3
decay_by = 0.005
lr_reduce_factor = 0.1#4*5e-2#1e-1#2e-2

batch_size = 512#1024#512#512#128#512#256

n_epochs = 11*2*int(14/number_of_trainingdata_sets)#21#21#21#20#20#15#7#5
n_its_per_epoch = 2**16#13#0#8#9#16 #used: min(len(c.train_loader), c.n_its_per_epoch)
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

x_dim = prepare_data.parameter_size #was 33/30, now 29# 18
y_dim_in = prepare_data.spectrum_size #1247 + 3 params+ 1
y_dim_features = 256#512#512#334#3066#384#512#167*2#256#int(740/5)#256#1253##1560#167#256#y_dim_in#148#256

# test_loader, train_loader = dataloader.get_loaders(batch_size)
test_ana_loader, train_ana_loader = dataloader.get_ana_loaders(batch_size)
#years = dc.viz_years


############
#  Losses  #
############

#loss_names = ['L_Train', 'L_Test']
#preview_upscale = 3

###########
#  Model  #
###########

n_blocks = 20#10#25#50#18#18#12
internal_width = int(256)#*2)#512
use_permutation = True
exponent_clamping = 1.0#7
init_scale = 0.03#6#75#0.075#1#0.03
fc_dropout = 0.#1#3#1#1#4#2#1#05#1#5#1#5#15#1#1#1#5#1#0.15#0.1 # not used at all
fn_dropout = 0.2#3#2#05#5#05#5#1#1#5#15#1#1#0.15#1 #feature net
skip_large = False

#####################
#  FeatureNet Setup #
#####################

solver = ["old","identity" ,"super" ,"cnn_solve" ,"cnn_long_solve","easy","cnn_sep_solve","Easy_solve","Harder_solve","Cnn_add_sep_solve"]
solver1 = ["seperate_hard_solve","Cnn_long_combined_solve", "fully_connected","FullyConnectedBig","Cnn_long_sep","Cnn_short_solve","PriorNet"]
feature_net_name = solver[4]#4]#12#15]#4]#10 #12 #13
#feature_net_name = solver1[6]#4]#12#15]#4]#10 #12 #13
print("Uses fn_model:",feature_net_name)

################
#  Evaluation  #
################

evaluation_samples = 1000#10000#90000#2000
N_post = 4096#512#
losses = []
sigma = []
mu = []
#params_mask = prepare_data.params_mask
#spec_delete = prepare_data.spec_delete

###########
#  Names  #
###########


configuration = f"{feature_net_name}"

import model
model = model.model
model.to(device)
