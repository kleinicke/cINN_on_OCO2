import numpy as np
import glob
import re

from pathlib import Path

"""Get required files from 
https://1drv.ms/f/s!AvXm21cCRGJ86hYwZC2TJvxV40Pk
and put them in co2_data/files.
Required by functions load_spects load_params
Returns:
    [type] -- [description]
"""
#additional_trainingdata_sets = [0,0,0,44,50,51,52,53,54]#0,0,0]
additional_trainingdata_sets = [0,0,0,50,51,52,53,54,55,56,57,58,59]#0,0,0]
#additional_trainingdata_sets = [0,0,0,0,0,0,0,0,0,0,0,0,0]#0,0,0]
additional_trainingdata_sets = [0]
#additional_trainingdata_sets = [44]#,50,51,52,53,54,55,56,57,58,59]#0,0,0]
#additional_trainingdata_sets = [50,51,52,53,54,55,56,57,58,59]#0,0,0]
#additional_trainingdata_sets = [0,0,44,50]
#additional_trainingdata_sets = [30,31,32]
#additional_trainingdata_sets = list(range(30,38))+[39]
number_of_additional_trainingdata_sets = len(additional_trainingdata_sets)
#testlist = list(range(10))+[43,52]+list(range(100,103))
#print("Testlist ",testlist)
log_spectra = True#False#True
kill_spectra = False
load_orig_data = True
external_config = 0 #9
use_additional_parameter = []+list(range(14))#list(range(14))+list(range(15,18))

skip_inputs = 14
training_config_text = "loads"
if load_orig_data:
    training_config_text += f" orig data and"
training_config_text += f"{number_of_additional_trainingdata_sets} additional sets. The sets are {additional_trainingdata_sets}"
training_config_text += f"spectra are "
training_config_text +=     "log" if log_spectra else "*e-10"
print(training_config_text)
mu_x,w_x,mu_y,w_y,mu_py,w_py=[0,0,0,0,0,0]
epsilon=1e-10
normizing_factors=[0.9e-3]
for i in range(14):
    normizing_factors.append(2.5e-3)
#normizing_factors.extend((1,4e-24,2e-20))
normizing_factors.extend((1,9e-5,0.5))
print(normizing_factors)
normizing_factors = normizing_factors[skip_inputs:]
for i in range(len(normizing_factors)):
    pass
    normizing_factors[i]=1

plotting=False

#if plotting:
#    import plot_data

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



def whitening_matrix(X,fudge=1e-7):
    '''https://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca'''

    print(f"real min/max of X:{np.amin(X),np.amax(X)}")
    # get the covariance matrix
    Xcov = np.cov(X.T)
    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)
    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    #print(f"min of d:{min(d)}")
    #fudge = 1e-20
    D = np.diag(1. / np.sqrt(d+fudge))
    #print(f"min/max of D:{min(min(D.tolist())),max(max(D.tolist()))}")
    W = np.dot(np.dot(V, D), V.T)
    Xw = np.dot(X, W)
    scaling = np.diag(1./np.std(Xw, axis=0))

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
    assert not np.isnan(np.sum(W))
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

def obs_to_y(observations):
    y1=observations[:,:-number_external_params]
    y2=observations[:,-number_external_params:]
    #print(np.shape(y1),np.shape(y2),np.shape(mu_y),np.shape(w_y),np.shape(mu_py),np.shape(w_py))
    #y = np.log(observations) done before
    y1 = np.dot(y1 - mu_y, w_y)
    y2 = np.dot(y2 - mu_py, w_py)
    #y1=np.zeros(np.shape(y1)) shading out all spectra for testing
    y=np.concatenate((y1,y2),axis=1)
    return y#np.clip(y, -5, 5)

def spectra_to_y(y):
    """deprecated, use obs_to_y(observations)
    """
    return obs_to_y(y)

def y_to_obs(y):
    obs1 = np.dot(y[:,:-number_external_params], np.linalg.inv(w_y)) + mu_y
    obs2 = np.dot(y[:,-number_external_params:], np.linalg.inv(w_py)) + mu_py
    if log_spectra:
        obs=np.concatenate((np.exp(obs1),obs2),axis=1)
    else:
        obs=np.concatenate(((obs1),obs2),axis=1)
    return obs

def y_to_spectra(y):
    """deprecated, use y_to_obs(y)
    """
    return y_to_obs(y)

def params_to_x(parameters,no_mu=False): 
    if no_mu:
        return np.dot(parameters, w_x)
    else:   
        return np.dot(parameters - mu_x, w_x)
    

def x_to_params(x,no_mu=False):
    if no_mu:
        return (np.dot(x, np.linalg.inv(w_x)) )/normizing_factors
    else:  
        return (np.dot(x, np.linalg.inv(w_x)) + mu_x)/normizing_factors


elements=[]
#for i in range(10,39):
for i in range(25,39):
    elements.append(i)
for i in [103,131,146,147]:
    elements.append(i)
#[external: range(10,13),pressure: range(13,26),temperature: range(26,39),103]
#print(elements)
#readSynth=True
element_names=[]
for i in range(1,14):
    #element_names.append(f"METEO_PRESSURE{i}")
    if i>12:
        element_names.append(f"Pressure {i}")
for i in range(1,14):
    #element_names.append(f"METEO_TEMPERATURE{i}")
    element_names.append(f"Temperature {i}")
#element_names.extend(("REMOTEC-OUTPUT_X_TRUE_COL","OUTPUT_X_STATE-ALB_WIN01_ORDER00","OUTPUT_X_APR_STATE-X_TYPE0001" , "OUTPUT_X_APR_STATE-X_TYPE0006"))
element_names.extend(("CO2","retrieved Albedo","H2O" , "CH4"))
external_names = ["SAT_ANGLE_SOLAR_ZENITH","SAT_ANGLE_VIEWING_ZENITH","SAT_RELATIVE_AZIMUTH_ANGLE"]
all_element_names = external_names
all_element_names.extend(element_names)
#print(f"external names: {np.shape(external_names)} {external_names}")
#print(f"element_names: {np.shape(element_names)} {element_names}")
print(f"all elements: {all_element_names}")
#TODO: Check if spect is matching
#TODO add external params to spect data. 
#external_elements=]
#for i in range(10,13):
#    external_elements.append(i)






def process_spects(external_params, path=Path(__file__).parent.joinpath("files/")): #,co2_mask
    global mu_y,w_y,mu_py,w_py
    #globpath = sorted(path.glob("raw_spectrum[1-9].npy"))
    print(f"Loads spectra from raw_spectrum0.npy")
    if load_orig_data:
        spectral_data = np.load(path.joinpath("raw_spectrum0.npy"))
    else:
        spectral_data = np.array([])
    #spectral_data_list=[]
    #spectral_data_list.append(spectral_data)
    #for i in range(10):
    #    print(external_params[i,:])
    #if number_of_trainingdata_sets > 1:
    #    print(f"Loads spectra from raw_spectrum1.npy")
    #    spectra = np.load(path.joinpath("raw_spectrum1.npy"))
    #    spectra=np.delete(spectra,7987, axis=0)
    #    spectra=np.delete(spectra,7946, axis=0)
    #    spectral_data= np.vstack([spectral_data, spectra])
    #    spectral_data_list.append(spectra)
    for i in additional_trainingdata_sets:
        #for i in globpath:
        print(f"Loads spectra from raw_spectrum{i}.npy")
        spectra= np.load(path.joinpath(f"raw_spectrum{i}.npy"))
        spectral_data= np.vstack([spectral_data, spectra]) if spectral_data.size else spectra
        #spectral_data_list.append(spectra)

    if kill_spectra:
        spectral_data[:,:] = 1e-13#spectral_data[:,0][:,None]
    print("Shape of all loaded spectra:",np.shape(spectral_data))
    print("and external params:",np.shape(external_params))
    #spectral_data=np.load(path)
    #spectral_data=spectral_data[co2_mask]
    if log_spectra:
        spectrum=np.log(spectral_data)
    else:
        spectrum=spectral_data*1e-11#np.asarray(spectral_data)[:,:,1]

    #spectral_data=np.delete(spectral_data,23508, axis=0)
    print("spectral_data: ", np.max(spectral_data),np.min(spectral_data))
    #print(np.sort(spectral_data,axis=1)[23507:23510,0:20])
    #print(np.unravel_index(spectral_data.argmin(), spectral_data.shape))
    #see delete in params:
    #if len(spectrum)>29000:
    #    spectrum=np.delete(spectrum,28394, axis=0)
    #    spectrum=np.delete(spectrum,28353, axis=0)

    #wavelenth=np.asarray(spectral_data)[:,:,0]
    wavelenth=np.load(path.joinpath("raw_wavelenth0.npy"))
    #print(np.shape(np.concatenate((np.log(longspect),external_params),axis=1)))

    #mu_y,w_y=(np.mean(np.log(spectrum), 0),whitening_matrix(np.log(spectrum)))
    mu_y,w_y=(np.mean((spectrum), 0),whitening_matrix((spectrum)))
    mu_py,w_py=(np.mean(external_params, 0),whitening_matrix(external_params))
    #print(external_params[0])
    #logspectrum=np.concatenate((np.log(spectrum),external_params),axis=1)
    logspectrum=np.concatenate(((spectrum),external_params),axis=1)
    wavelenth=np.concatenate(((wavelenth[:len(external_params)]),external_params[:len(wavelenth)]),axis=1)

    #print(logspectrum[0,-number_external_params:])
    y=obs_to_y(logspectrum)
    #y_list = []
    #for i,spec in enumerate(spectral_data_list):
    #    spec2 = np.concatenate((np.log(spec),e_param_list[i]),axis=1)
    #    y_list.append(obs_to_y(spec2))
    #print(y[0,-number_external_params:])
    print(not np.isnan(np.sum(logspectrum)))
    assert not np.isnan(np.sum(logspectrum))
    return logspectrum,wavelenth, y#, y_list


def progress_params(path=Path(__file__).parent.joinpath("files/")):
    global mu_x, w_x,element_names
    #globpath = sorted(path.glob("raw_parameters[1-9].npy"))

    if load_orig_data:
        print(f"Loads params from raw_parameters0.npy")
        all_params = np.load(path.joinpath("raw_parameters0.npy"))
    else:
        all_params = np.array([])
    orig_data_size = len(all_params)
    #param_list=[]
    #param_list.append(all_params)
    #if number_of_trainingdata_sets > 1:
    #    print(f"Loads params from raw_parameters1.npy")
    #    params= np.load(path.joinpath("raw_parameters1.npy"))
    #    #What is wrong with these two values? 15 is way to high.
    #    print(params[7900:8000,131])
    #    print(params[7945:7947,:])
    #    params=np.delete(params,7987, axis=0)
    #    params=np.delete(params,7946, axis=0)
    #   #print(params[7900:8000,131])
    #   param_list.append(params)
    #   all_params= np.vstack([all_params, params])


    for i in additional_trainingdata_sets:

        #for i in globpath:
        print(f"Loads params from raw_parameters{i}.npy")
        params= np.load(path.joinpath(f"raw_parameters{i}.npy"))
        #all_params.append(params)
        print(f"params{i} ",np.shape(params),f"params so far ",np.shape(all_params))
        all_params= np.vstack([all_params, params]) if all_params.size else params
        #param_list.append(params)

    if not load_orig_data:
        orig_data_size = len(all_params)
   
    #all_params = np.asarray(all_params)
    print("Shape of all loaded params:",np.shape(all_params))


    #if len(all_params)>28394:
    #    all_params=np.delete(all_params,28394, axis=0)
    #    all_params=np.delete(all_params,7946, axis=0)


    #getting rid of co2 outlier
    #co2_mask=np.logical_and(np.greater_equal(all_params[:,103],377),np.less(all_params[:,103],389))
    #print("elements before getting rid of outlying co2 values",np.shape(co2_mask),np.shape(params))
    #all_params=all_params[co2_mask]
    #print("elements after getting rid of outlying co2 values",np.shape(co2_mask),np.shape(params))


    output_params = all_params[:,elements]
    print(np.max(output_params))
    print(np.unravel_index(np.argmax(output_params, axis=None), output_params.shape))
    #pressure_params=params[:,13:26]
    #temp_params=params[:,26:39]
    external_params=all_params[:,[3,8,9,10]] # Month, Latitude, (Longitude), Solar Angle
    #print("\n\nTest1:\n",all_params[:,[3,8,9,10]])
    #for i in range(5):
    #    print(external_params[i])
    sin_month=np.sin(external_params[:,0]/12*2*np.pi)
    cos_month=np.cos(external_params[:,0]/12*2*np.pi)
    sin_lat=np.sin(external_params[:,1]/90*np.pi)
    cos_lat=np.cos(external_params[:,1]/90*np.pi)
    sin_long=np.sin(external_params[:,2]/160*np.pi)
    cos_long=np.cos(external_params[:,2]/160*np.pi)
    
    #determines 5 external parameter
    external_params=np.stack([sin_month*100,cos_month*100,external_params[:,1],external_params[:,2],external_params[:,3]],axis=1)
    if external_config == 0:
        print("external configuration 0")
    elif external_config == 1:
        external_params = np.ones(np.shape(external_params))
        print("external configuration 1, month, position and sun")
    elif external_config == 2:
        external_params = external_params[:,:2]
        print("external configuration 2, month only")
    elif external_config == 3:
        external_params = external_params[:,2:4]
        print("external configuration 3, position only")
    elif external_config == 4:
        external_params = np.stack([all_params[:,10],all_params[:,10]],axis=1)
        print("external configuration 4, sun only")
    elif external_config == 5:
        external_params = np.stack([external_params[:,0],external_params[:,1],all_params[:,10]],axis=1)
        print("external configuration 5, month and sun")
    elif external_config == 6:
        external_params = np.stack([external_params[:,2],external_params[:,3],all_params[:,10]],axis=1)
        print("external configuration 6, pos and sun")
    elif external_config == 7:
        external_params = np.stack([all_params[:,3],all_params[:,10]],axis=1)
        print("external configuration 4, sun only")
    elif external_config == 8:
        external_params = np.stack([all_params[:,4],all_params[:,10]],axis=1)
        print("external configuration 4, sun only")

    else:
        print("external configuration 9")
        external_params=all_params[:,[3,8,9,10]]
    #external_params = np.stack([external_params[:,0],external_params[:,1],all_params[:,10]],axis=1)
    """
    for i in range(10):
        print(all_params[i*1000,[3,8,9,10]])
    for i in range(10):
        print(all_params[i,[3,8,9,10]])
    for i in range(10):
        print(all_params[i+orig_data_size,[3,8,9,10]])
    """
    #assert 0
    #output_params=output_params[:,4:]
    #output_params=output_params[:,11:]
    #element_names=element_names[12:]

    
    #calculates CH4 to ppm instead of molecules per cubic centimeter
    # CH4/METEO_AIRMASS_INTEGRATED
    output_params[:,-1]=(output_params[:,-1]/all_params[:,64]*1e6)
    output_params[:,-2]=(output_params[:,-2]/all_params[:,64]*1e6)

    #adds all parameters to inputs. 
    #for i in range(17):
    #    if i>13:
    #        external_params=np.concatenate((external_params,output_params[:,i+1][:,None]),axis=1)
            #external_params=np.concatenate((external_params,external_params[:,0][:,None]),axis=1)
    #        pass
    #    else:
    #        external_params=np.concatenate((external_params,output_params[:,i][:,None]),axis=1)
    #        pass

    for i in use_additional_parameter:
        #print(np.shape(external_params))
        #print(np.shape(output_params[:,i]))
        external_params=np.concatenate((external_params,output_params[:,i][:,None]),axis=1)
    print("Total number of external params",np.shape(output_params))

    #print(np.shape(external_params),np.shape(output_params[:,0]))

    #print(np.shape(external_params))
    #print("New Params:")
    #print(external_params)

    #CO2 retrieval errors
    co2_ret_params = all_params[:,99:104]
    co2_ret_params = np.concatenate((co2_ret_params,all_params[:,[3,8,9,10]]),axis=1)

    t_p_params = output_params[:,:skip_inputs]
    output_params = output_params[:,skip_inputs:]
    output_params=output_params*normizing_factors
    #for i in list(range(14))+list(range(15,18)):
    #    output_params[:,i]=1
    for i in range(len(output_params[0])):
        print(i,np.amin(output_params[:,i]),np.amax(output_params[:,i]))

    """
    print(np.amax(output_params,axis=1))
    print(np.amax(output_params))
    print(np.shape(output_params))
    print(np.argmax(output_params))
    print(np.unravel_index(np.argmax(output_params, axis=None), output_params.shape))
    print(output_params)
    print(output_params[28352:28355,:])
    print("\n")
    for i in range(300):
        start=0
        maximum = (np.amax(output_params[start+i*100:start+i*100+100,15]))
        print(maximum)
        if maximum > 1:
            print(start+i*100)
            print(output_params[start+i*100:start+i*100+100,15])
            print(np.unravel_index(np.argmax(output_params[start+i*100:start+i*100+100],axis=None), output_params.shape))

    print("\n")
    print("\n")
    """
    mu_x,w_x=(np.mean(output_params, 0),whitening_matrix(output_params,fudge=1e-50))#stddev_matrix(longparams))
    w_x = stddev_matrix(output_params)
    print("mu_x",mu_x)
    print("w_x",w_x)
    output_x=params_to_x(output_params)
    #TODO whats the differentce between ret_params rows and train_x...
    print(not np.isnan(np.sum(output_params)))
    assert not np.isnan(np.sum(output_params))
    #print(co2_ret_params[:10,2])
    #print(x_to_params(output_x)[:10,14])
    #print(all_params[:10,103])
    #x_list = []
    #for i in param_list:
        #print("param_list",param_list)
        #print(np.shape(i))
    #    i[:,elements[-1]]=i[:,elements[-1]]/i[:,64]*1e6
    #    i[:,elements[-2]]=i[:,elements[-2]]/i[:,64]*1e6
    #    i=i[:,elements]
    #    i=i*normizing_factors
    #    print("normalized params:",np.max(i),np.min(i))
    #    x_list.append(params_to_x(i))
        #print("normalized x:",(x_list),np.min(x_list))
    
    #e_param_list = []
    #for i in param_list:
    #    e_params = i[:,[3,8,9,10]]
    #    e_params = np.stack([np.sin(e_params[:,0]/12*2*np.pi)*100,np.cos(e_params[:,0]/12*2*np.pi)*100,e_params[:,1],e_params[:,2],e_params[:,3]],axis=1)
    #    e_param_list.append(e_params)

    return output_params,external_params, co2_ret_params, output_x, orig_data_size,t_p_params #,co2_mask, x_list, e_param_list,


#loads parameter and spectrum
output_params,external_params,co2_ret_params,output_x, orig_params_size,t_p_params = progress_params() #co2_mask
number_external_params = np.shape(external_params)[1]

output_spects,wavelenth_y,output_y = process_spects(external_params)#,co2_mask)
parameter_size,spectrum_size = np.shape(output_x[:,:])[1],np.shape(output_y)[1]

print(f"param_size: {parameter_size}, spect_size {spectrum_size}")
print(f"shape of whitened {np.shape(output_y)} and original specturm {np.shape(output_spects)} and of the wavelenth {np.shape(wavelenth_y)}")








if plotting:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt 
    plot_data.plot()
    plt.show()