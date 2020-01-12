import torch
import os
import numpy as np
import data.prepare_data as prepare_data
import data.data_config as dc
#import viz
import __main__
main_file = __main__.__file__.split("/")[-1][:-3]

train_uncert = True
longer = False

params_in_spectrum = prepare_data.params_in_spectrum #5
number_of_spectra = 3
spec_length = prepare_data.spec_length
print(f"number of external parameter is {params_in_spectrum}")
co2_pos = 0
param_names = prepare_data.params_names
ana_names = prepare_data.ana_names
element_names = list(prepare_data.namedict.keys())
run_name = "feature"
#if len(param_names)>9:
#    arbitrary_number = 1
#elif len(param_names)>6:
#    arbitrary_number = 2
#else:
# 2,3,4 is uncert 
arbitrary_number = 5 if train_uncert else 1
print(len(param_names))
######################
#  General settings  #
######################

#load_file = 'output/co2INN1_18.pt_checkpoint_0480_2'
load_file = ''
#number_of_trainingdata_sets = prepare_data.number_of_additional_trainingdata_sets + 1
#if number_of_trainingdata_sets > 1:
#    add1 = prepare_data.additional_trainingdata_sets[0]
#else:
#    add1 = ""
#data = f"{number_of_trainingdata_sets}_{add1}_"
#data = ""
load_checkpoit = 0#30
load_checkpoit = 0#5#"5"
filename = f'output/co2INN{arbitrary_number}.pt'+"1"
if load_checkpoit:
    filename += f"_checkpoint_{load_checkpoit:04d}"
#filename = 'output/co2INN1_28.pt_checkpoint_0064'

print("File name is",filename)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device is {device}")

#assert device == 'cuda'
#When cuda isn't active anymore for some reason restart it with the commands:
"""sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm
"""


live_visualization = True
progress_bar = True

checkpoint_save_interval = 5#64#120 
checkpoint_save_overwrite = False
#######################
#  Training schedule  #
#######################
lr_reducer = 3*1#*1e-1
lr_init = 5e-4*lr_reducer#1e-3
decay_by = 0.005

lr_reduce_factor = 5e-2#1e-1#2e-2

batch_size = 1024#2048#2048#512#512#128#512#256

n_epochs = 11#30#20#20#15#7#5
n_its_per_epoch = 2**10#13#0#8#9#16 #used: min(len(c.train_loader), c.n_its_per_epoch)
do_rev = False


l2_weight_reg = 1e-5
adam_betas = (0.9, 0.999)

pre_low_lr =1 #15
latent_noise = 1.

blockTestData = True
stop_train_loss=1e30

#####################
#  Data dimensions  #
#####################

import data.dataloader as dataloader

x_dim = (dataloader.parameter_size) #was 33/30, now 29# 18, 6
y_dim_in = dataloader.spectrum_size #1247 + 3 params+ 1, 3064
y_dim_features = 512#512#334#3066#384#512#167*2#256#int(740/5)#256#1253##1560#167#256#y_dim_in#148#256

# test_loader, train_loader = dataloader.get_loaders(batch_size)
test_ana_loader, train_ana_loader = dataloader.get_ana_loaders(batch_size)
year_sets = dataloader.year_sets
#test_loader = train_loader
#test_ana_loader# =  train_ana_loader
############
#  Losses  #
############

loss_names = ['L_Train', 'L_Test']
preview_upscale = 3
predict_uncert = True


#####################
#  FeatureNet Setup #
#####################

#use_pretrained = True
#additionally_trained_feature_net=False#True#False#Falselif additionally_trained_feature_net:

#new_Featurenet = True
#if new_Featurenet:
#    use_pretrained = False
#    additionally_trained_feature_net = False

feature_net_file = f'output/featureNet_{arbitrary_number}.pt'
solver = ["old","identity" ,"super" ,"cnn_solve" ,"cnn_long_solve","easy","cnn_sep_solve","Easy_solve","Harder_solve","Cnn_add_sep_solve"]
solver2 = ["seperate_hard_solve","Cnn_long_combined_solve", "fully_connected","FullyConnectedBig","Cnn_long_sep","Cnn_short_solve","Seperate_easy_solve","old_seperate_hard_solve"]

feature_net_name = solver2[0]#12#15]#4]#10 #12 #13
#feature_net_name = solver2[7]#12#15]#4]#10 #12 #13
print("Uses fn_model:",feature_net_name,feature_net_file)

load_model = False
if load_model:
    load_file = filename#[:-1]
    additionally_trained_feature_net = True
    filename = filename+"+"#[:-1]

    #additionally_trained_feature_net = True
#else:
#    load_file = "output/defaultFN.pt"

save_new_fn=True

#fn_train_lr=1e-6*lr_reducer#4e-6#5#6#4#9
#fn_pretrain_lr = 7e-2
update_fn_params = True
start_featurenet_train = -pre_low_lr# first epoch w.o. featurenet training or -1
#y_dim_features=params_in_spectrum

#################
#  FN Pretrain  #
#################
fn_dropout = 0.2#2#05#5#05#5#1#1#5#15#1#1#0.15#1 #feature net

fn_pretrain_percentage = n_epochs
fn_pretrain_log_interval = 10
fn_pretrain_number_of_tests = 1#00#50

##############
#  FN Train  #
##############

################
#  Evaluation  #
################

evaluation_samples = 1024#512#20002048#
losses = []
sigma = []
mu = []
params_mask = prepare_data.params_mask
spec_delete = prepare_data.spec_delete

####
#Names
###

configuration = f"{main_file[:5]}_{feature_net_name}"

import nets
#if longer:
model = nets.get_model(outputs = 2 * x_dim) if train_uncert else nets.get_model()
#else:
#    model = nets.get_model(outputs = 2) if train_uncert else nets.get_model()
