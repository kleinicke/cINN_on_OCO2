import numpy as np
import torch
import torch.utils.data
from os.path import join
import config as c
import math

#import os
#current_dir = os.path.dirname(os.path.abspath(__file__))
from pathlib import Path

try:
    import data.prepare_data as prepare_data
except:
    import prepare_data as prepare_data


#x_file = f'{current_dir}/files/co2_data_parameters.npy'
#y_file = f'{current_dir}/files/co2_data_observations.npy'
##x_file = Path(__file__).parent.joinpath("files/co2_data_parameters.npy")
##y_file = Path(__file__).parent.joinpath("files/co2_data_observations.npy")
##print(x_file)
##all_x = np.load(x_file)
##all_y = np.load(y_file)

all_x = prepare_data.output_x[:]
all_y = prepare_data.output_y[:]
orig_data_size = prepare_data.orig_params_size

#test_split = int(0.2 * orig_data_size)
#test_split -= (test_split%c.batch_size)
test_split = 1000
test_split += c.batch_size-(test_split%c.batch_size)

print('Training set size with orig data:\t', orig_data_size - test_split)
print('Test set size:\t\t', test_split, '\n')
#train_x,train_y = [all_x[test_split:], all_y[test_split:]]
#test_x,test_y = [all_x[:test_split], all_y[:test_split]]

np.random.seed(999)
perm = np.random.permutation(orig_data_size)
test_x = all_x[:orig_data_size][perm][:test_split]
test_y = all_y[:orig_data_size][perm][:test_split]
#np.random.choice(aa_milne_arr, 5

#external_params = prepare_data.external_params
#test_externals = external_params[:orig_data_size][perm][:test_split]
t_p_params = prepare_data.t_p_params
test_output = t_p_params[:orig_data_size][perm][:test_split]
#mask = np.ones(len(all_x), dtype = int)
#math.isclose()
all_same = np.ones(len(all_x), dtype = bool)
print(np.sum(all_same))
print("\n")
for i in test_output:
    #print(np.shape(i),np.shape(test_x))
    #close_mask = np.allclose(i[:1],test_x[:,:1], atol = 1e-1)
    #print(np.shape(close_mask))
    #print(close_mask)
    comp = 14
    off = 0
    diff = np.abs(t_p_params[:,off:off+comp]-i[off:off+comp])
    diff = np.sum(diff, axis = 1)
    same = diff < 1e-6 # True if found in test
    #print(same)
    #print(np.sum(same))
    all_same= np.logical_xor(all_same,same)

print("Total training set size: \t",np.sum(all_same))
print(f"Kicked out {len(all_x)-np.sum(all_same)}/{len(all_x)} values. {(len(all_x)-np.sum(all_same))/test_split} values per testdata. (expected >={prepare_data.number_of_additional_trainingdata_sets+int(prepare_data.load_orig_data)}")
print(np.shape(all_same),np.shape(all_x),np.shape(all_y))
train_x,train_y = [all_x[all_same], all_y[all_same]]

"""
mask = np.ones(len(all_x), dtype=bool)
print(mask)
print(perm)
perm = perm[mask]
perm_mask = [0 if i<test_split else 1 for i in perm]
print(perm,"\n")
#print(perm_mask,"\n")
print(np.sum(perm_mask))
print("x",all_x)
print("y",all_y)

assert 0
mask = np.ones(len(all_x), dtype = int)
mask[:,test_split] = 0
perm_inv = np.argsort(perm)
mask = mask[perm_inv]
exist_mask = np.ones(len(all_x), dtype=bool)
train_mask = mask[exist_mask]
"""
print(not np.isnan(np.sum(all_x)))
assert not np.isnan(np.sum(all_x))
print(not np.isnan(np.sum(all_y)))
assert not np.isnan(np.sum(all_y))
#perm_test = np.random.permutation(test_split)
#perm_train = np.random.permutation(len(all_x)-test_split)
#test_x,test_y = [test_x[perm_test],test_y[perm_test]]
#train_x,train_y = [train_x[perm_train],train_y[perm_train]]

#ret_params=prepare_data.co2_ret_params[perm]
ret_params=prepare_data.co2_ret_params[:orig_data_size][perm][:test_split]
#for i in range(10):
#    print("test6,",i,ret_params[i,-4:])
#external_params=prepare_data.co2_ret_params[perm][:test_split]
#np.concatenate((y1,ret_params),axis=1)
#ret_test = ret_params[:test_split][perm_test]
#ret_train = ret_params[test_split:][perm_train]
#ret_params = np.concatenate((ret_test,ret_train),axis=0)
#print(np.shape(ret_test),np.shape(ret_train))

#for i in range(20):
#    print(f"gt_co2: {prepare_data.x_to_params(all_x[:30])[i,14]}, ret params: {ret_params[i]}")
# To see if you can fit the prior data distribution:
# all_y *= 0.
#print(ret_params[:10,2])
#print(prepare_data.x_to_params(test_x)[:10,14])

#all_x = all_x[perm]
#all_y = all_y[perm]
all_x = torch.Tensor(all_x)
all_y = torch.Tensor(all_y)
test_x,test_y = torch.Tensor(test_x), torch.Tensor(test_y)
train_x,train_y = torch.Tensor(train_x), torch.Tensor(train_y)
#ana_1,ana_2 = torch.Tensor(test_x), torch.Tensor(test_y)


print(f"Total size of Dataset: x:{np.shape(all_x)}, y:{np.shape(all_y)}")


def get_ana_loaders(batch_size, seed=0):
    torch.manual_seed(seed)
    test_loader = torch.utils.data.DataLoader(
        #torch.utils.data.TensorDataset(all_x[:test_split], all_y[:test_split]),
        torch.utils.data.TensorDataset(test_x,test_y,test_x),
        batch_size=batch_size, shuffle=False, drop_last=True, num_workers= 6)

    train_loader = torch.utils.data.DataLoader(
        #torch.utils.data.TensorDataset(all_x[test_split:], all_y[test_split:]),
        torch.utils.data.TensorDataset(train_x,train_y,train_x),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers= 6)

    return test_loader, train_loader
