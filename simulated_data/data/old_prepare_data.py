import numpy as np
import glob
import re

#import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(current_dir)
#Start with: [13:26],[26:39],[10:13],[103],[146,147],[131]

#REMOTEC-OUTPUT_X_APR_STATE X_TYPE0001 , X_TYPE0006 [146,147]

#REMOTEC-OUTPUT_X_STATE ALB_WIN01_ORDER00 [131]
"""Get required files from 
https://1drv.ms/f/s!AvXm21cCRGJ86hYwZC2TJvxV40Pk
and put them in co2_data/files.
Required by functions load_spects load_params
Returns:
    [type] -- [description]
"""

mu_x,w_x,mu_y,w_y,mu_py,w_py=[0,0,0,0,0,0]
epsilon=1e-10
normizing_factors=[0.9e-3]
for i in range(14):
    normizing_factors.append(2.5e-3)
#normizing_factors.extend((1,4e-24,2e-20))
normizing_factors.extend((1,9e-5,0.5))
print(normizing_factors)

plotting=False

if plotting:
    import plot_data

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
    print(np.shape(X))
    #print(f"min/max of X:{min(min(X.tolist())),max(max(X.tolist()))}")
    print(f"real min/max of X:{np.amin(X),np.amax(X)}")
    # get the covariance matrix
    if plotting:
        X_tmp=X[:,14:]
        #X_tmp=np.delete(X,15,1)
        #X_tmp=np.delete(X_tmp,15,1)
        #np.concatenate(X_tmp,X[:,17:])
        #X_tmp.append(X[:,17:])
        print(np.shape(X_tmp),np.shape(max(X.tolist())))
        Xcov = np.cov(X.T)
        #TODO Xcov is wrong

        #for i in range(18):
        #    print(i )
            #print(min(X.tolist())[i],max(X.tolist())[i])
        #    print(np.amin(X,axis=0)[i],np.amax(X,axis=0)[i])
        #˘›    print("")
        #for i in X[:,-1]:
        #    print(i)
        #print(min(X[:,-1]),max(X[:,-1]))
        #print(np.amin(X,axis=0))
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt 
        plt.figure()
        plt.imshow(Xcov)
        plt.colorbar()
        #plt.figure("max_min")
        #plt.plot(range(np.shape(X)[1]),max(X.tolist()),label="max")
        #plt.plot(range(np.shape(X)[1]),min(X.tolist()),label="min")
        #plt.legend()
        plt.figure()
        plt.plot(range(np.shape(X)[1]),np.amin(X,axis=0),label="min")
        plt.plot(range(np.shape(X)[1]),np.amax(X,axis=0),label="max")
        plt.legend()
        #plt.show()
        #assert False

    Xcov = np.cov(X.T)
    # eigenvalue decomposition of the covariance matrix
    d, V = np.linalg.eigh(Xcov)
    # a fudge factor can be used so that eigenvectors associated with
    # small eigenvalues do not get overamplified.
    print(f"min of d:{min(d)}")
    D = np.diag(1. / np.sqrt(d+fudge))
    #print(f"min/max of D:{min(min(D.tolist())),max(max(D.tolist()))}")
    W = np.dot(np.dot(V, D), V.T)
    Xw = np.dot(X, W)
    scaling = np.diag(1./np.std(Xw, axis=0))
    #print("from x to xw whitened ",np.shape(X),np.shape(Xw),np.shape(scaling))

    return np.dot(W, scaling)

def svd_whiten(X):

    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    X_white = np.dot(U, Vt)

    return X_white

def stddev_matrix(X):
    return np.diag(1./(np.std(X, axis=0)+epsilon))

#mu_x, w_x, mu_y, w_y = (np.mean(parameters, 0),
#                        stddev_matrix(parameters),
#                        np.mean(np.log(observations), 0),
#                        whitening_matrix(np.log(observations)))

def obs_to_y(observations):
    y=observations
    #y = np.log(observations) done before
    y = np.dot(y - mu_y, w_y)
    return np.clip(y, -5, 5)

def y_to_obs(y):
    obs = np.dot(y, np.linalg.inv(w_y)) + mu_y
    return np.exp(obs)

def params_to_x(parameters):
    return np.dot(parameters - mu_x, w_x)
    #np.dot(parameters, np.linalg.inv(w_x)) + mu_x
    

def x_to_params(x):
    return (np.dot(x, np.linalg.inv(w_x)) + mu_x)/normizing_factors


elements=[]
for i in range(10,39):
    elements.append(i)
for i in [103,131,146,147]:
    elements.append(i)
#[external: range(10,13),pressure: range(13,26),temperature: range(26,39),103]
print(elements)
#readSynth=True
element_names=[]
for i in range(1,14):
    #element_names.append(f"METEO_PRESSURE{i}")
    element_names.append(f"PRESSURE{i}")
for i in range(1,14):
    #element_names.append(f"METEO_TEMPERATURE{i}")
    element_names.append(f"TEMPERATURE{i}")
#element_names.extend(("REMOTEC-OUTPUT_X_TRUE_COL","OUTPUT_X_STATE-ALB_WIN01_ORDER00","OUTPUT_X_APR_STATE-X_TYPE0001" , "OUTPUT_X_APR_STATE-X_TYPE0006"))
element_names.extend(("CO2","retrieved Albedo"," H2O" , "CH4"))
external_names = ["SAT_ANGLE_SOLAR_ZENITH","SAT_ANGLE_VIEWING_ZENITH","SAT_RELATIVE_AZIMUTH_ANGLE"]
all_element_names = external_names
all_element_names.extend(element_names)
print(f"external names: {np.shape(external_names)} {external_names}")
print(f"element_names: {np.shape(element_names)} {element_names}")
print(f"all elements: {all_element_names}")
#TODO: Check if spect is matching
#TODO add external params to spect data. 
#external_elements=]
#for i in range(10,13):
#    external_elements.append(i)


def save_spect_set(longspect,path=f"{current_dir}/files"):
    #mu, sigma = 0, 0.00001 
    # creating a noise with the same dimension as the dataset (2,2) 
    #noise = np.random.normal(mu, sigma, np.shape(longspect))    
    #print(noise)
    #print(longspect)
    #longspect=longspect+noise
    longspect=obs_to_y(longspect)

    #print("spect,params",np.shape(longspect),np.shape(longparams))
    #print("longspect",np.shape(longspect))

    #special testset not required. While loading test/train set is created
    ##test_spect=longspect[:1000,:] #(0,1) shows position and value
    ##train_spect=longspect[1000:,:]
    ##np.save(f"{path}/test_y_spec.npy",test_spect)
    ##np.save(f"{path}/train_y_spec.npy",train_spect)
    np.save(f"{path}/co2_data_observations.npy",longspect)
    

    return longspect


def save_param_set(params,path=f"{current_dir}/files"):
    #mu, sigma = 0, 0.0001 
    # creating a noise with the same dimension as the dataset (2,2) 
    #noise = 1#-np.random.normal(mu, sigma, np.shape(params))    
    #print(noise)
    #print(params)
    reducedlong=params_to_x(params*noise)
    #np.save(f"{path}/test_x_param.npy",reducedlong[:1000,:])
    #np.save(f"{path}/train_x_param.npy",reducedlong[1000:,:])
    np.save(f"{path}/co2_data_parameters.npy",reducedlong[:,:])
    return reducedlong

#save_params()
#save_spect()

def load_spects(new_params,co2_mask,path=f"{current_dir}/files/raw_spectrum.npy"):
    global mu_y,w_y
    spect=np.load(path)
    spect=spect[co2_mask]
    longspect=np.asarray(spect)[:,:,1]
    wavelenth=np.asarray(spect)[:,:,0]
    print(np.shape(np.concatenate((np.log(longspect),new_params),axis=1)))
    #print(np.stack([np.log(longspect),new_params],axis=1))
    #assert 0
    logspect=np.concatenate((np.log(longspect),new_params),axis=1)
    wavelenth=np.concatenate((np.log(wavelenth),new_params),axis=1)
    mu_y,w_y=(np.mean(logspect, 0),whitening_matrix(logspect))
    return logspect,wavelenth

def load_params(path=f"{current_dir}/files/raw_parameters.npy"):
    global mu_x, w_x,element_names
    params= np.load(path)
    co2_mask=np.logical_and(np.greater_equal(params[:,103],377),np.less(params[:,103],389))
    #print("elements before getting rid of outlying co2 values",np.shape(co2_mask),np.shape(params))
    params=params[co2_mask]
    #print("elements after getting rid of outlying co2 values",np.shape(co2_mask),np.shape(params))
    longparams = params[:,elements]
    pressure_params=params[:,13:26]
    temp_params=params[:,26:39]
    new_params=params[:,[3,8,9,10]] # Month, Latitude, (Longitude) Solar Angle
    sin_month=np.sin(new_params[:,0]/12*2*np.pi)
    cos_month=np.cos(new_params[:,0]/12*2*np.pi)
    sin_lat=np.sin(new_params[:,1]/90*np.pi)
    cos_lat=np.cos(new_params[:,1]/90*np.pi)
    sin_long=np.sin(new_params[:,2]/160*np.pi)
    cos_long=np.cos(new_params[:,2]/160*np.pi)
    circle_params=np.stack([sin_month,cos_month,sin_lat,cos_lat,sin_long,cos_long],axis=1)
    new_params=np.stack([sin_month*100,cos_month*100,new_params[:,1],new_params[:,2],new_params[:,3]],axis=1)
    
    #print("circle shape",np.shape(circle_params),np.shape(new_params))
    #uselesslist=[0,3]
    #externallist=[1,2]
    useless_params=longparams[:,0:4]#always the same
    #external=longparams[:,externallist]#always the same
    longparams=longparams[:,4:]
    #correlated=longparams[:,:11]#barometric hight profile starting from one value
    longparams=longparams[:,11:]
    element_names=element_names[12:]

    #print(f"64 {params[:,64]}, 63 {params[:,63]}, 65 {params[:,65]}\n 66 {params[:,66]} 67 {params[:,67]}")
    
    #normalizes params
    #print("\n PPM PARAMS:")
    longparams[:,-1]=(longparams[:,-1]/params[:,64]*1e6)
    longparams[:,-2]=(longparams[:,-2]/params[:,64]*1e6)
    #print(np.min(longparams[:,-2]/params[:,64]*1e6),np.max(longparams[:,-2]/params[:,64]*1e6),np.min(longparams[:,-1]/params[:,64]*1e6),np.max(longparams[:,-1]/params[:,64]*1e6))
    for i in range(18):
        #if i>13:
        #    new_params=np.concatenate((new_params,longparams[:,i+1][:,None]),axis=1)
        #else:
            new_params=np.concatenate((new_params,longparams[:,i][:,None]),axis=1)

    longparams=longparams*normizing_factors
    print(np.shape(new_params),np.shape(longparams[:,0]))

    #adds all parameters to inputs. Should be really easy to solve
    for i in range(18):
        #if i>13:
        #    new_params=np.concatenate((new_params,longparams[:,i+1][:,None]),axis=1)
        #else:
            new_params=np.concatenate((new_params,longparams[:,i][:,None]),axis=1)

    print(np.shape(new_params))
    print("New Params:")
    print(new_params)

    print("CO2 retrieval errors")
    co2_ret_params=params[:,99:104]
    print(co2_ret_params)
    #print(params[:,103])
    #longparams[:,-2]=longparams[:,-2]*1e-22#20
    #longparams[:,-1]=longparams[:,-1]*1e-17
    #longparams[:,-3]=longparams[:,-3]*1e4
    #longparams[:,-4]=longparams[:,-4]*1e-3
    #longparams[:,0]=longparams[:,0]*1e-3

    #fudge=1e-12
    #longparams+=fudge
    #print(min(min(longparams.tolist())))
    mu_x,w_x=(np.mean(longparams, 0),whitening_matrix(longparams,fudge=1e-50))#stddev_matrix(longparams))
    return longparams,useless_params, pressure_params, temp_params,new_params,circle_params,co2_mask, co2_ret_params

lists=[50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100]
good_lists=[50,150,200,250,300,350,550,600,650,700,850,1000,1050,1100]
#loads parameter and spectrum
params,useless_params, pressure_params, temp_params,new_params,circle_params,co2_mask,co2_ret_params=load_params()
print(np.shape(params))
#print(external_params[:100,:])
#print(useless_params[:100,:])
spects,wavelenth=load_spects(new_params,co2_mask)
print(np.shape(spects))
#puts external parameters that shoulnd't be guessed into spectrum
#spects=np.append(spects,params[:,:3],axis=1)
#params=params[:,3:]
print(f"size of spects {np.shape(spects)} and params {np.shape(params)}")
#whighening and saving of parameter and spectra
x_long = save_param_set(params)
print(np.shape(x_long))
y_long = save_spect_set(spects)
retreved = y_to_obs(y_long)
retreved2 = y_to_obs(y_long[0:5,:])
#y_long=np.append(y_long,params[:,:3],axis=1)
parameter_size,spectrum_size=np.shape(x_long[:,:])[1],np.shape(y_long)[1]
print(f"param_size: {parameter_size}, spect_size {spectrum_size}")
print(f"shape of whitened {np.shape(y_long)} and original specturm {np.shape(spects)} and of the wavelenth {np.shape(wavelenth)}")

#wparam=whightened_param(param)
#print(f"param: {param[0,:]}, wparam {wparam[0,:]}")
#print(mu_x)
#wspect=save_spect_set(spect)
#print(f"spect: {spect[0,:]}, wspect {wspect[0,:]}")

#parameter_size=30
#spectrum_size=1250







if plotting:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt 
    plot_data.plot()
    plt.show()