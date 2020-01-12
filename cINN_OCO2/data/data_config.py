"""Configurations for the dataloader and datapreprocessing
"""
import numpy as np

#select years to test and train
train_years = [2014, 2015, 2016, 2017, 2018]
#train_years = [2014]
test_years = [2019]
#noise of the data
train_noise_sigma = 0#10*1e-5#4#4e-4#4e-2

#Use smaller or bigger dataset
short_ds = True
if short_ds:
    short_str = "_short"
else:
    short_str = ""

#sample size for visualization datasets
viz_size = 512#1024#2048#1024#512

#is dataset already normalized? -> 1, else 2
preprocessing_mode = 1

additional_noise = True

viz_years = ['2014','2015','2016','2017','2018','2019']
#viz_years = ['2014']
sum_of_years = False

params_mask = np.array([0,0,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0],dtype=bool)#9#10
use_spectra = True
use_params = True

#these positions in spectra will be deleted. Make it positive to delete nothing.
#spec_delete = np.array([-(len(spectra_names)-a_index)])#10
spec_delete = np.array([5])

#Set true once when data is changed
create_new_norm = True
