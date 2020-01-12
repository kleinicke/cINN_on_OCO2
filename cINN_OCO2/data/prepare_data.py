import numpy as np
import glob
import re

from pathlib import Path
import time
import timeit
import math
import copy
from copy import deepcopy as dcopy

import matplotlib.pyplot as plt
try:
    import data.data_config as dc
except:
    import data_config as dc



#1 means precalculating the normalization
#2 means normalizing within the dataloader
#########prepare_data Settings ######
spec_length= 1016
kernel_length = 20
spec_start = 34+1+kernel_length
#################################


namedict = {
    "L2_id":0,
    "xco2":1,
    "albedo_o2":2,
    "albedo_sco2":3,
    "albedo_wco2":4,
    "tcwv":5,
    "aod_bc":6,
    "aod_dust":7,
    "aod_ice":8,
    "aod_oc":9,
    "aod_seasalt":10,
    "aod_sulfate":11,
    "aod_total":12,
    "aod_water":13,
    "t700":14,
    "psurf":15,
    "windspeed":16,
    "sza":17,
    "latitude":18,
    "longitude":19,
    "year":20,
    "month":21,
    "day":22,
    "sensor_zenith_angle":23,
    "snr_wco2":24,
    "snr_sco2":25,
    "snr_o2a":26,
    "glint_angle":27,
    "altitude":28,
    "tcwv_apriori":29,
    "tcwv_uncertainty":30,
    "xco2_apriori":31,
    "xco2_uncertainty":32,
    "xco2_raw":33,
    "xco2_averaging_kernel":list(range(34,34+kernel_length)),
    "L1_id":34+kernel_length,
    "sco2":list(range(spec_start,spec_start+spec_length)),
    "wco2":list(range(spec_start+spec_length,spec_start+spec_length*2)),
    "o2":list(range(spec_start+spec_length*2,spec_start+spec_length*3)),
}

#spectra is input of nn, params is output of nn, ana is used to analyze the results later
spectra_names = ["sco2","wco2","o2","snr_wco2","snr_sco2","snr_o2a","sza","day","month","month","month","year","xco2_apriori","tcwv_apriori","sensor_zenith_angle","glint_angle","altitude","psurf","t700","longitude","latitude"] 
params_names = np.array(["xco2","xco2","xco2","xco2","albedo_o2","albedo_sco2","albedo_wco2","tcwv","tcwv","aod_bc","aod_dust","aod_ice","aod_oc","aod_seasalt","aod_sulfate","aod_total","aod_water"])
#First real entries: "xco2_apriori","xco2_raw","xco2-xco2_apriori","xco2",
ana_names = ["xco2","tcwv","tcwv_apriori","tcwv_uncertainty","xco2_apriori","xco2_uncertainty","xco2_raw","xco2_averaging_kernel","longitude","latitude"]

#select relevant parameter for params and spectra
params_mask = dc.params_mask
#np.array([1,1,0,1,1,1,1,0,1,0,0,0,0,0,0,0,0],dtype=bool)#9#10
#params_mask = np.array([0,1,1,1,1,0,1,0,0,0,0,0,0,0,0],dtype=bool)#9#10
#params_mask = np.array([0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],dtype=bool)#9#10


#params_mask = np.array([1,1,1,1,1,0,1,1,0,0,1,1,1],dtype=bool)
#params_mask = np.array([1,1,1,1,1,0,1,0,0,0,0,1,0],dtype=bool)
#params_mask = np.array([1,0,1,1,1,1,0,0,0,0,0,0,0,1,0],dtype=bool)#8#11
#params_mask = np.array([1,1,1,1,1,1,0,0,0,0,0,0,0,1,0],dtype=bool)#13
params_names = params_names[params_mask]
print(params_names)

spec_mask = np.ones(len(spectra_names),dtype=bool)
a_index = spectra_names.index("xco2_apriori")

#these positions in spectra will be deleted. Make it positive to delete nothing.
#spec_delete = np.array([-(len(spectra_names)-a_index)])#10
#spec_delete = np.array([5])
spec_delete = dc.spec_delete
####spec_mask[a_co2_offset] = False

print(np.delete(np.asarray(spectra_names),len(spectra_names)+spec_delete))
print(np.asarray(spectra_names))
params_in_spectrum = len(spectra_names)-3


#positions of names of spec and analysis
spectra_pos = []
for spec in spectra_names:
    spec = namedict[spec]
    spectra_pos+= spec if type(spec) is list else [spec]
params_pos = [namedict[params] for params in params_names]

ana_pos = []
for ana in ana_names:
    ana = namedict[ana]
    ana_pos+= ana if type(ana) is list else [ana]




#relevant folder
#here the samples for normization are stored
samples_path = Path(__file__).parent.joinpath("files/1_sp_samples/")
#results of normazation are stroed here
normed_path = Path(__file__).parent.joinpath("files/2_normed")
#in this folder the unnormalized data was stored in subfolder 4_spectraparams_{year} 
#is normalized and stored in the subfolder 4_xy_{year}{dc.short_str}
files_folder_path = Path(__file__).parent.joinpath("files/")

def normalize_data():
    """Loades sampled data to determine the whitening parameter for them.
    This function has to be called not often. The print statements are nessesary to evaluate the quality of the outcome.
    """

    samples_p = sorted(samples_path.glob(f"sample_params*.npy"))
    samples_s = sorted(samples_path.glob(f"sample_spectra*.npy"))
    print(samples_p,samples_s)
    sample_params = np.load(samples_p[-1])
    sample_spectra = np.load(samples_s[-1])#[:,:105]
    print("min_param",sample_params.min(axis=0))
    print("max_param",sample_params.max(axis=0))
    print("Loaded samplesize of ",sample_params.shape,sample_spectra.shape,params_mask.shape,spec_delete)
    sample_params = sample_params[:,params_mask]
    print("min_param",sample_params.min(axis=0))
    print("max_param",sample_params.max(axis=0))
    sample_spectra = np.delete(sample_spectra,len(sample_spectra[0])+spec_delete,axis=1)
    print(sample_spectra[0,:10])
    print("Filtered samplesize of ",(sample_params.shape),(sample_spectra.shape))
    print("w_x")
    mu_x,w_x = (np.mean(sample_params, 0),whitening_matrix(sample_params))
    print("max_param:",np.max(np.dot(sample_params - mu_x, w_x)),np.min(np.dot(sample_params - mu_x, w_x)))
    w_dev_x, w_dev_y = stddev_matrix(sample_params), stddev_matrix(sample_spectra)
    print("max_param_dev:",np.max(np.dot(sample_params - mu_x, w_dev_x)),np.min(np.dot(sample_params - mu_x, w_dev_x)))

    print("w_y")
    mu_y,w_y = (np.mean(sample_spectra, 0),whitening_matrix(sample_spectra))#130 seconds
    re = np.dot(sample_spectra - mu_y, w_y)
    print("mu_x_y", mu_x,mu_y)
    print("w_x_y", w_x,w_y)
    print("w_dev_x_y", w_dev_x, w_dev_y)


    print(re[:10])
    print("1:",np.max(re,axis=1))
    print("1:",np.argmax(re,axis=1))
    print("0:",np.max(re,axis=0))
    print("0:",np.argmax(re,axis=0))

    obs = np.dot(re[:,:], np.linalg.inv(w_y)) + mu_y
    print(np.max(obs),np.min(obs))
    print(obs[:2])
    print(sample_spectra[:2])

    print("w_dev_y")
    w_dev_x, w_dev_y = stddev_matrix(sample_params), stddev_matrix(sample_spectra)

    #w_x = w_dev_x
    x_s = np.dot(sample_params - mu_x, w_dev_x)
    params = (np.dot(x_s, np.linalg.inv(w_x)) + mu_x)
    
    print("min_param",x_s.min(axis=0))
    print("max_param",x_s.max(axis=0))
    print("min_param",params.min(axis=0))
    print("max_param",params.max(axis=0))
    print("w_y123")
    w_y1, w_y2, w_y3 = whitening_matrix(sample_spectra[:,:spec_length]), whitening_matrix(sample_spectra[:,spec_length:2*spec_length]), whitening_matrix(sample_spectra[:,2*spec_length:3*spec_length])
    w_dev_y4, w_y4 = stddev_matrix(sample_spectra[:,3*spec_length:]), whitening_matrix(sample_spectra[:,3*spec_length:])
    print("mu_w_x", mu_x,w_x)
    print("mu_w_y", mu_y,w_y)
    print("w_dev_x_y", w_dev_x, w_dev_y)
    np.save(normed_path.joinpath("mu_x.npy"),mu_x)
    np.save(normed_path.joinpath("w_x.npy"),w_x)
    np.save(normed_path.joinpath("w_y1.npy"),w_y1)
    np.save(normed_path.joinpath("w_y2.npy"),w_y2)
    np.save(normed_path.joinpath("w_y3.npy"),w_y3)
    np.save(normed_path.joinpath("w_y4.npy"),w_y4)
    np.save(normed_path.joinpath("w_dev_y4.npy"),w_dev_y4)
    np.save(normed_path.joinpath("mu_y.npy"),mu_y)
    np.save(normed_path.joinpath("w_y.npy"),w_y)
    np.save(normed_path.joinpath("w_dev_x.npy"),w_dev_x)
    np.save(normed_path.joinpath("w_dev_y.npy"),w_dev_y)

def load_elems():
    """Loads the normalization parameter. These can be used to normalize and whiten the trainingdata
    
    Returns:
        mu_y, w_y, mu_x, w_x, w_dev_x, w_dev_y, w_y1, w_y2, w_y3, w_y4, w_dev_y4
    """
    print("Loads normalization terms")
    mu_x = np.load(normed_path.joinpath("mu_x.npy"))
    w_x = np.load(normed_path.joinpath("w_x.npy"))
    mu_y = np.load(normed_path.joinpath("mu_y.npy"))
    w_y = np.load(normed_path.joinpath("w_y.npy"))
    w_dev_x = np.load(normed_path.joinpath("w_dev_x.npy"))
    w_dev_y = np.load(normed_path.joinpath("w_dev_y.npy"))
    w_y1, w_y2, w_y3, w_y4 = np.load(normed_path.joinpath("w_y1.npy")), np.load(normed_path.joinpath("w_y2.npy")), np.load(normed_path.joinpath("w_y3.npy")), np.load(normed_path.joinpath("w_y4.npy"))
    w_dev_y4 = np.load(normed_path.joinpath("w_dev_y4.npy"))
    return mu_y, w_y, mu_x, w_x, w_dev_x, w_dev_y, w_y1, w_y2, w_y3, w_y4, w_dev_y4



def whitening_matrix(X,fudge=1e-5):
    """
    determines whitening parameter of given X-input data  
    as done in https://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca

    Arguments:
        X {np.ndarray} -- contains spectra or parameter. Pay attention, that they range within similar magnitudes.
    
    Keyword Arguments:
        fudge {float} -- for stability (default: {1e-5})
    
    Returns:
        w -- Whitening matrix. Use this to scale the parameters, after its mean is subtracted.
    """


    print(f"real min/max of X for whitening creation:{np.amin(X),np.amax(X)}")
    t_white = time.perf_counter()

    # get the covariance matrix
    Xcov = np.cov(X.T)

    print("Xcov",Xcov)
    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)
    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    #print(f"min of d:{min(d)}")
    #fudge = 1e-20
    D = np.diag(1. / np.sqrt(d+fudge))
    print("D",D)
    #print(f"min/max of D:{min(min(D.tolist())),max(max(D.tolist()))}")
    W = np.dot(np.dot(V, D), V.T)
    Xw = np.dot(X, W)
    scaling = np.diag(1./np.std(Xw, axis=0))
    print("scaling",scaling)
    print("Xw",Xw,np.max(Xw))
    print("W",W,np.max(W))
    
    #print("W",W)
    #print("scaling",scaling)
    #print(X)
    #print(Xcov)
    #print(d)
    #print("\n")
    #print("d:",d) # this is interesting when nan occured during sqrt
    #print(np.sqrt(d+fudge))
    #print("")
    #print(V)
    #print(D)
    #print("\n")
    #print(np.dot(V, D))
    #print(V.T)
    print(f"whiteing took {time.perf_counter()-t_white:.2f} s.")
    assert not np.isnan(np.sum(W))

    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.subplot(2,2,1)
    #plt.imshow(Xcov, vmin=0, vmax=1)
    #plt.subplot(2,2,2)
    #plt.imshow(np.log(Xcov))
    #plt.colorbar()
    #plt.subplot(2,2,3)
    #plt.imshow(np.log(1. - Xcov))
    #plt.colorbar()
    #plt.subplot(2,2,4)
    #plt.imshow(np.dot(W, scaling))
    #plt.colorbar()
    #plt.show()
    
    
    return np.dot(W, scaling)

def stddev_matrix(X):
    """Alternative function to whitening_matrix. Returns a matrix which is only filled in the diagonal.
    
    Arguments:
        X {np.ndarray} -- contains spectra or parameter. Pay attention, that they range within similar magnitudes.
    
    Returns:
        w -- Weight matrix. Use this to scale the parameters, after its mean is subtracted.
    """
    epsilon=1e-10
    return np.diag(1./(np.std(X, axis=0)+epsilon))



#use this when new normalizationdata is required
if dc.create_new_norm:
    normalize_data()

#loads most recent normalization data
mu_y, w_y, mu_x, w_x, w_dev_x, w_dev_y, w_y1, w_y2, w_y3, w_y4, w_dev_y4 = load_elems()
#switch the use of some of these values here, so they don't have to be renamed below.
#w_y4 = w_dev_y4
#w_y = w_dev_y
w_x = w_dev_x
#w_y4 = w_dev_y4
#w_y = w_dev_y
#w_x = w_dev_x
#assert 0

print("\n\n")
print("mu_x \n", mu_x)
print("w_x\n", w_x)
print("mu_y\n", mu_y)
print("w_y\n", w_y)
print("w_dev_x\n", w_dev_x)
print("w_dev_y\n", w_dev_y)
print("\n\n")

def spectra_to_y(observations, mu_y = mu_y, w_y = w_y, no_norm = False ,mult_w_y = False, w_y1 = w_y1, w_y2 = w_y2, w_y3 = w_y3, w_y4 = w_y4):
    """Normalize the spectra with the given parameter
    
    Arguments:
        observations {[np.ndarray]} -- spectra
    
    Keyword Arguments:
        mu_y {[np.ndarray]} --  [shifts mean to 0] (default: {mu_y})
        w_y {[np.ndarray]} -- [scaling] (default: {w_y})
        no_norm {bool} -- [True for no scaling] (default: {False})
        mult_w_y {bool} -- [False for don't use below values] (default: {False})
        w_y1 {[np.ndarray]} -- [scaling for first spectrum] (default: {w_y1})
        w_y2 {[np.ndarray]} -- [scaling for second spectrum] (default: {w_y2})
        w_y3 {[np.ndarray]} -- [scaling for third spectrum] (default: {w_y3})
        w_y4 {[np.ndarray]} -- [scaling for additional parameter] (default: {w_y4})
    
    Returns:
       y [np.ndarray] -- [normalized input for neural network]
    """

    y=observations
    np.log(observations[:,:3*spec_length], out = observations[:,:3*spec_length])
    
    #should scaling be applied?
    if no_norm:
        y = y - mu_y
        print("no_norm", y.max(), y.min())
    else:
        #should a different scaling be applied on different parts of the set? 1,2 and 3rd spectrum and external parameter
        if mult_w_y:
            y1 = np.dot(y[:,:spec_length] - mu_y[:spec_length], w_y1)
            y2 = np.dot(y[:,spec_length:2*spec_length] - mu_y[spec_length:2*spec_length], w_y2)
            y3 = np.dot(y[:,2*spec_length:3*spec_length] - mu_y[2*spec_length:3*spec_length], w_y3)
            y4 = np.dot(y[:,3*spec_length:] - mu_y[3*spec_length:], w_y4)
            y = np.concatenate((y1,y2,y3,y4),axis=1)
            print("y_sep", y.max(), y.min())
        else:
            #is dataset already normalized?
            #if dc.preprocessing_mode == 2:
            #    y = np.subtract(y, mu_y)
            #else:
                #default case: applies one whitening matrix on all the data
            y = np.dot(y - mu_y, w_y)

            
    return y

def y_to_spectra(y, mu_y = mu_y, w_y = w_y, mult_w_y = False, w_y1 = w_y1, w_y2 = w_y2, w_y3 = w_y3, w_y4 = w_y4):
    """Turns input data of the Neural Network back to data that can be analyzed. 
    (f.e. to read out positions of the sample that is stored at the two last positions)
    
    Arguments:
        y {[np.ndarray]} -- normalized spectrum and additional parameters
    
    Keyword Arguments:
        mu_y {[np.ndarray]} -- [description] (default: {mu_y})
        w_y {[np.ndarray]} -- [description] (default: {w_y})
        mult_w_y {bool} -- [False for don't use below values] (default: {False})
        w_y1 {[np.ndarray]} -- [scaling for first spectrum] (default: {w_y1})
        w_y2 {[np.ndarray]} -- [scaling for second spectrum] (default: {w_y2})
        w_y3 {[np.ndarray]} -- [scaling for third spectrum] (default: {w_y3})
        w_y4 {[np.ndarray]} -- [scaling for additional parameter] (default: {w_y4})
    
    Returns:
        [np.ndarray] -- readable spectrum and additional parameters
    """
    if mult_w_y:
        obs1 = np.dot(y[:,:spec_length], np.linalg.inv(w_y1)) + mu_y[:spec_length]
        obs2 = np.dot(y[:,spec_length:2*spec_length], np.linalg.inv(w_y2)) + mu_y[spec_length:2*spec_length]
        obs3 = np.dot(y[:,2*spec_length:3*spec_length], np.linalg.inv(w_y3)) + mu_y[2*spec_length:3*spec_length]
        obs4 = np.dot(y[:,3*spec_length:], np.linalg.inv(w_y4)) + mu_y[3*spec_length:]
        obs = np.concatenate((obs1,obs2,obs3,obs4),axis=1)
    else:
        #is dataset already normalized?
        #if dc.preprocessing_mode == 2:
        #    obs = np.add( y[:,:] , mu_y)
        #else:
            #default case
        obs = np.dot(y[:,:], np.linalg.inv(w_y)) + mu_y
    #the spectra had been brought down from 1e16 to 1e20 range with log function.
    #this brings it back up inplace.
    obs=np.concatenate((np.exp(obs[:,:-params_in_spectrum]),obs[:,-params_in_spectrum:]),axis=1)
    return obs

def params_to_x(parameters, mu_x = mu_x, w_x = w_x,no_mu=False):
    """Normalizes the parameter so they can better be processed by Neural Network
    
    Arguments:
        parameters {[np.ndarray]} -- [unnormalized parameter]
    
    Keyword Arguments:
        mu_x {[np.ndarray]} -- [to change mean value to 0] (default: {mu_x})
        w_x {[np.ndarray]} -- [scaling parameter] (default: {w_x})
    
    Returns:
        x [np.ndarray] -- [normalized parameter]
    """
    assert not mu_x is 0
    assert not w_x is 0
    if no_mu:
        return np.dot(parameters, w_x)
    else:   
        return np.dot(parameters - mu_x, w_x)    

def x_to_params(x, mu_x = mu_x, w_x = w_x,no_mu=False):
    """Make output of NN understandable again
    
    Arguments:
        x [np.ndarray] -- [normalized parameter]
    
    Keyword Arguments:
        mu_x {[np.ndarray]} -- [to change mean value to 0] (default: {mu_x})
        w_x {[np.ndarray]} -- [scaling parameter] (default: {w_x})

    Returns:
        parameters {[np.ndarray]} -- [unnormalized parameter]
    """
    if no_mu:
        return (np.dot(x, np.linalg.inv(w_x)) )
    else:  
        return (np.dot(x, np.linalg.inv(w_x)) + mu_x)



def apply_norm():
    """Applies normalization on data and saves x and y.
    Only needs to be called, once per dataset.
    """
    ##mu_x = np.load(normed_path.joinpath("mu_x.npy"))
    ##w_x = np.load(normed_path.joinpath("w_x.npy"))
    #w_x = np.load(normed_path.joinpath("w_dev_x.npy"))
    ##mu_y = np.load(normed_path.joinpath("mu_y.npy"))
    ##w_y = np.load(normed_path.joinpath("w_y.npy"))
    #w_y = np.load(normed_path.joinpath("w_dev_y.npy"))
    year_folders = sorted(files_folder_path.glob(f"4_spectraparams_*_short/"))
    print(year_folders)
    #year_folders = [Path('files/4_spectraparams_2018')]
    t_norm_all = time.perf_counter()
    print(year_folders)
    for year in year_folders:
        year = str(year)[-4-6:-6]
        print(year)
        t_norm = time.perf_counter()

        p_list = sorted(files_folder_path.glob(f"4_spectraparams_{year}{dc.short_str}/params*.npy"))
        s_list = sorted(files_folder_path.glob(f"4_spectraparams_{year}{dc.short_str}/spectra*.npy"))
        ana_list = sorted(files_folder_path.glob(f"4_spectraparams_{year}{dc.short_str}/ana*.npy"))
        for i in range((len(p_list))):
            number = str(p_list[i]).split('/')[-1:][0].strip()[7:10]
            number2 = str(s_list[i]).split('/')[-1:][0].strip()[8:11]
            number3 = str(ana_list[i]).split('/')[-1:][0].strip()[4:7]

            print(number, len(p_list), year)
            assert number == number2
            assert number == number3
            
            params = np.load(p_list[i])[:,params_mask]
            spectra = np.load(s_list[i])
            spectra = np.delete(spectra,len(spectra[0])+spec_delete,axis=1)
            ana = np.load(ana_list[i])
            assert not np.isnan(np.sum(spectra))    

            x = params_to_x(params)#,mu_x, w_x)
            y = spectra_to_y(dcopy(spectra))#,mu_y, w_y)

            
            assert not np.isnan(np.sum(x))
            assert not np.isnan(np.sum(y))       


            np.save(files_folder_path.joinpath(f"4_xy_{year}{dc.short_str}/x_{number}"),x)
            np.save(files_folder_path.joinpath(f"4_xy_{year}{dc.short_str}/y_{number}"),y)
            np.save(files_folder_path.joinpath(f"4_xy_{year}{dc.short_str}/ana_{number}"),ana)
        print(f"normalizing {year} took {time.perf_counter()-t_norm:.3f}s or {(time.perf_counter()-t_norm)/60:.3f}minutes.")
    print(f"normalizing all took {time.perf_counter()-t_norm_all:.3f}s or {(time.perf_counter()-t_norm_all)/60:.3f}minutes.")

#apply_norm to normalize given dataset. Only needs to be called once when set is changed.
if dc.create_new_norm:
    apply_norm()


