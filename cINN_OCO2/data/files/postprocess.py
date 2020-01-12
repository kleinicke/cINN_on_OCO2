"""Further filters down the data used for the Neural Network.
"""

import numpy as np
import glob
import re

from pathlib import Path
import time
import timeit
import math



spec_length= 1016
kernel_length = 20
spec_start = 34+1+kernel_length
#parameter to extract and their position in the array
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

#the names of the parameter that are extracted to a certain position
spectra_names = ["sco2","wco2","o2","snr_wco2","snr_sco2","snr_o2a","sza","day","month","month","month","year","xco2_apriori","tcwv_apriori","sensor_zenith_angle","glint_angle","altitude","psurf","t700","longitude","latitude"] 

params_names = np.array(["xco2_apriori","xco2_raw","xco2","xco2","albedo_o2","albedo_sco2","albedo_wco2","tcwv","tcwv","aod_bc","aod_dust","aod_ice","aod_oc","aod_seasalt","aod_sulfate","aod_total","aod_water"])
params_names_list = list(params_names)

ana_names = ["xco2","tcwv","tcwv_apriori","tcwv_uncertainty","xco2_apriori","xco2_uncertainty","xco2_raw","xco2_averaging_kernel","longitude","latitude"]

#calculates the positions of the read parameter, since some parameter require multiple positions
spectra_pos = []
for spec in spectra_names:
    spec = namedict[spec]
    spectra_pos+= spec if type(spec) is list else [spec]
params_pos = [namedict[params] for params in params_names]

ana_pos = []
for ana in ana_names:
    ana = namedict[ana]
    ana_pos+= ana if type(ana) is list else [ana]

#for colored process outputs
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



#results from the process.py file
filtered_L1L2_path = Path(__file__).parent.joinpath("../generate/npy")

#paths to important folders where subresults are stored
#if they don't exist yet, create the folders at the given path and copy data from npy folder to 0 folder
filtered_L1L2_path = Path(__file__).parent.joinpath("0_filtered_L1L2")
#the resulting spectra and params are stored here
spectraparams_path = Path(__file__).parent.joinpath("1_spectraparams/")
#for normalization some samples will be created
samples_path = Path(__file__).parent.joinpath("1_sp_samples/")
#the overall folder
files_folder_path = Path(__file__).parent#.joinpath("../files/")



def sort_elems():
    """
    Resorts the files into parameter and spectra files which could be used for input and output by the neural network.
    Throws out bad parameter and spectra and creates a small random subsample for normalization of the data.
    The files afterwards have to be processed to be in equally sized chunks to be used be the dataloader.

    Returns:
        Nothing -- But saves spectra, params, ana_elems, sample_spectra and sample_params
    """

    #samples params and spectra to calculate normalization factors, since all data is to much to be used for this.
    sample_params = np.array([])
    sample_spectra = np.array([])

    t_total = time.perf_counter()
    print("Start with samplesize of ",len(sample_params))
    period = "9"#"[4-9]"
    L2s = sorted(filtered_L1L2_path.glob(f"L2_1{period}*.npy"))
    L1s = sorted(filtered_L1L2_path.glob(f"L1_1{period}*.npy"))
    print("L2s",L2s)
    print("L1s",L1s)
    number_of_files = 20 #5
    for i in range(math.ceil(len(L2s)/number_of_files)):
        L2_names = L2s[i*number_of_files:(i+1)*number_of_files]
        L1_names = L1s[i*number_of_files:(i+1)*number_of_files]
        week = str(L2_names[0]).split('/')[-1:][0].strip()[3:9]
        print(f"\n\n\n Starting week {week}\n")
        t_month = time.perf_counter()
        L1_elems = np.array([])
        L2_elems = np.array([])

        for path in L2_names:#sorted(filtered_L1L2_path.glob(f"L2_{week}*.npy")):
            L2 = np.load(str(path),mmap_mode = 'c')
            L2_elems = np.concatenate([L2_elems, L2],axis=0) if len(L2_elems) > 0 else L2
            print(f"L2 filelenth: {len(L2_elems)}")

        for path in L1_names:#sorted(filtered_L1L2_path.glob(f"L1_{week}*.npy")):
            L1 = np.load(str(path),mmap_mode = 'c')
            L1_elems = np.concatenate([L1_elems, L1],axis=0) if len(L1_elems) > 0 else L1
            print(f"L1 filelenth: {len(L1_elems)}")

        print(np.shape(L1_elems),np.shape(L2_elems))

        #create one big file
        all_elems = np.concatenate([L2_elems, L1_elems],axis=1) 
        print(len(all_elems))
        filtering = True
        if filtering:
            #lon and lat are stored in the last two positions of spectra
            all_positions = all_elems[:,spectra_pos[-2:]]
            #filter out very dense data in south africa and australia
            #todo: antifilter less common regions. y > 55 or y < -55
            #todo: second level of common: 36 > y > 25, -115 < x < 96; iran, sahara, maybe use map to visualize it
            pos_mask = np.logical_and( all_positions[:,0] > 0 , all_positions[:,1] < -10)
            length_mask = np.full(len(all_elems), False)
            length_mask[:int(len(all_elems)*0.8)] = True
            np.random.shuffle(length_mask)
            pos_mask = np.logical_and(length_mask,pos_mask)
            
            print(np.sum(pos_mask))
            
            #reduce number of samples in general
            length_mask = np.full(len(all_elems), False)
            length_mask[:int(len(all_elems)/30)] = True
            np.random.shuffle(length_mask)
            length_mask =  np.logical_and(length_mask, np.logical_not(pos_mask)) 
            print(np.sum(pos_mask))
            print("shape before",all_elems.shape)

            all_elems=all_elems[length_mask]
            print("shape after reduction",all_elems.shape)
        selecting = False
        if selecting:
            #selecting only spain
            # -9.55 < long < 3.16 and 36.9 < lat < 42.43
            all_positions = all_elems[:,spectra_pos[-2:]]
            pos_mask = np.logical_and( all_positions[:,0] > -9.55 , all_positions[:,0] < 3.16)
            pos_mask2 = np.logical_and( all_positions[:,1] > 36.9 , all_positions[:,1] < 42.43)
            pos_mask = np.logical_and(pos_mask,pos_mask2)
            print("before filtering",all_elems.shape)
            all_elems=all_elems[pos_mask]
            print("after filtering",all_elems.shape)
            
        #show range of values of each parameter. Afterwards filter nans and unwanted values
        for i in range(len(list(namedict))):
            print("elems",i,list(namedict)[i],np.min(all_elems[:,namedict[list(namedict)[i]]]),np.max(all_elems[:,namedict[list(namedict)[i]]])) #todo add name of elem
 
        params = all_elems[:,params_pos]
        spectra = all_elems[:,spectra_pos]


        #filter unwanted values
        print(f"found NaN values in params: {np.sum(np.isnan(params))} and spectra: {np.sum(np.isnan(spectra))}")
        print(f"faulti spectra: {np.sum(np.isnan(spectra).any(axis=1))}")
        print(f"faulti channel: {np.sum(np.isnan(spectra).any(axis=0))}")
        print(np.isnan(spectra).any(axis=0))
        print(np.shape(np.isnan(spectra).any(axis=0)))
        print("small t:",np.sum(spectra[:,-1]<-10000))
        print("small 668:",np.sum(spectra[:,668]<1e10))
        print("small spectra:",np.sum((spectra[:,:3*spec_length]<1e5).any(axis=1)))
        print("small both:",np.sum(np.logical_or(spectra[:,668]<1e10,spectra[:,-1]<-10000)))
        unusual_mask = np.logical_and(np.logical_not(np.isnan(spectra).any(axis=1)),np.logical_not((spectra[:,:3*spec_length+3]<1e0).any(axis=1)))#np.logical_and(unusual_mask,)
        print("deletes",len(spectra)-np.sum(unusual_mask),"spectra")
        print("len params, spectra before",len(params),len(spectra))
        params = params[unusual_mask]
        spectra = spectra[unusual_mask]

        all_elems = all_elems[unusual_mask]
        print("applied masks")

        #show range of cleaned parameter
        for i in range(3):
            print(spectra_names[i],i,np.min(spectra[:,i*spec_length:(i+1)*spec_length]),np.max(spectra[:,i*spec_length:(i+1)*spec_length])) 
        
        ana_elems = all_elems[:,ana_pos]
        #modify a few parameter, to store more valuable information inside
        #f.e. substract apriori from value or add month and day to year.
        #These are stored in dublicated or unnessesary spots.
        i_apriori = ana_names.index("xco2_apriori")
        i_xco2 = params_names_list.index("xco2")
        params[:,i_xco2] -= ana_elems[:,i_apriori]
        i_tcwv_apriori = ana_names.index("tcwv_apriori")
        i_tcwv = np.where(params_names=="tcwv")[0][0]
        i_month = spectra_names.index("month")
        i_rel_month = i_month - len(spectra_names)
        i_rel_day = i_rel_month -1
        day_noise = np.random.normal(0,1/30,spectra[:,i_rel_day].shape)
        spectra[:,i_rel_month+1]=np.sin((spectra[:,i_rel_month]+spectra[:,i_rel_day]/30+day_noise)/12*2*np.pi)
        spectra[:,i_rel_month+2]=np.cos((spectra[:,i_rel_month]+spectra[:,i_rel_day]/30+day_noise)/12*2*np.pi)

        spectra[:,i_rel_day] = spectra[:,i_rel_month] + spectra[:,i_rel_day]/30 + spectra[:,i_rel_month+3]*12
        spectra[:,i_rel_day] += day_noise

        params[:,i_tcwv] -=ana_elems[:,i_tcwv_apriori]
        np.save(spectraparams_path.joinpath(f"spectra{week}.npy"),spectra)
        np.save(spectraparams_path.joinpath(f"params{week}.npy"),params)
        np.save(spectraparams_path.joinpath(f"ana_elems{week}.npy"),ana_elems)

        #Sample some spec and elems to apply whightening on the data
        #None is used to free storage
        all_elems = None
        ana_elems = None
        print(params[0:2,:10])

        print("len params, spectra after",len(params),len(spectra))
        print("all spectra",np.min(spectra[:,:]),np.max(spectra[:,:]))

        #inplace log function of part of the array
        np.log(spectra[:,:3*spec_length],out = spectra[:,:3*spec_length])
        logspectra = spectra
        spectra = None

        #logspectra should have a good quality. Say no nans.
        print("all logspectra",np.min(logspectra[:,:]),np.max(logspectra[:,:]),np.shape(logspectra)) 
        print("found nans in logspectrum:",np.isnan(logspectra).any())

        #sampling
        sampled_size = 100 
        choosen_spectra = logspectra[np.random.choice((logspectra.shape[0]),size = sampled_size, replace = False)]
        sample_spectra = np.concatenate([sample_spectra, choosen_spectra],axis=0) if len(sample_spectra) > 0 else choosen_spectra
        choosen_params = params[np.random.choice((params.shape[0]),size = sampled_size, replace = False)]
        sample_params = np.concatenate([sample_params, choosen_params],axis=0) if len(sample_params) > 0 else choosen_params
        print("samples:", np.shape(sample_params),np.shape(sample_spectra))


        assert not np.isnan(np.sum(params))
        assert not np.isnan(np.sum(logspectra))

        np.save(samples_path.joinpath(f"sample_spectra{week}.npy"),sample_spectra)
        np.save(samples_path.joinpath(f"sample_params{week}.npy"),sample_params)



        print(f"Time process month: {time.perf_counter() - t_month:.2f}s")
    print(f"Time process all month: {time.perf_counter() - t_total:.2f}s")




def sort_arrays(year="2015", filesize = 1000):
    """Rearranges arrays, to be equally sized, so loading is easiert for the dataloader
    """

    params_list = sorted(spectraparams_path.glob(f"params{year[-2:]}*.npy"))#[0-5]
    spectra_list = sorted(spectraparams_path.glob(f"spectra{year[-2:]}*.npy"))
    ana_list = sorted(spectraparams_path.glob(f"ana_elems{year[-2:]}*.npy"))
     
    print(params_list)
    print(spectra_list)
    print(ana_list)
    params_values = np.array([])
    spectra_values = np.array([])
    ana_values = np.array([])
    index = 0
    for i in range((len(params_list))):
        number = str(params_list[i]).split('/')[-1:][0].strip()[6:12]
        number2 = str(spectra_list[i]).split('/')[-1:][0].strip()[7:13]
        print(i, number,number2)
        assert number == number2, print(number,number2)
        params_val = np.load(params_list[i])
        spectra_val = np.load(spectra_list[i])       
        ana_val = np.load(ana_list[i])
        params_values = np.concatenate([params_values, params_val],axis=0) if len(params_values) > 0 else params_val
        spectra_values = np.concatenate([spectra_values, spectra_val],axis=0) if len(spectra_values) > 0 else spectra_val
        ana_values = np.concatenate([ana_values, ana_val],axis=0) if len(ana_values) > 0 else ana_val
        print(len(params_values))
        assert len(params_values)==len(spectra_values),f"{len(params_values),len(spectra_values)}"
        assert len(ana_values)==len(params_values),f"length{len(ana_values),len(params_values)}"
        mod = "_short"
        while len(params_values)>filesize:
            np.save(files_folder_path.joinpath(f"4_spectraparams_{year}{mod}/params_{index:03d}"),params_values[:filesize])
            np.save(files_folder_path.joinpath(f"4_spectraparams_{year}{mod}/spectra_{index:03d}"),spectra_values[:filesize])
            np.save(files_folder_path.joinpath(f"4_spectraparams_{year}{mod}/ana_{index:03d}"),ana_values[:filesize])
            
            index+=1
            params_values = np.delete(params_values, np.s_[0:filesize], axis = 0)
            spectra_values = np.delete(spectra_values, np.s_[0:filesize], axis = 0)
            ana_values = np.delete(ana_values, np.s_[0:filesize], axis = 0)
            print(np.shape(params_values))

def shorten_test(filesize = 10000):
    """Can shorten the existing dataset. This is usefull f.e. to create a subdataset afterwards (f.e. for a smaller testset or smaller datasets in general). 
    This functionality is now standard in the sort_elems function anyways.
    
    Keyword Arguments:
        filesize {int} -- [description] (default: {10000})
    """
    x_list = sorted(xy_test_path.glob(f"x*.npy"))
    y_list = sorted(xy_test_path.glob(f"y*.npy"))
    ana_list = sorted(xy_test_path.glob(f"ana*.npy"))
    x_values = np.array([])
    y_values = np.array([])
    ana_values = np.array([])
    for i in range(len(x_list)):
        x = np.load(x_list[i])
        y = np.load(y_list[i])       
        ana = np.load(ana_list[i])
        mask = np.full(len(x), False)
        mask[:int(len(x)/20)] = True
        np.random.shuffle(mask)
        x=x[mask]
        y=y[mask]
        ana = ana[mask]
        x_values = np.concatenate([x_values, x],axis=0) if len(x_values) > 0 else x
        y_values = np.concatenate([y_values, y],axis=0) if len(y_values) > 0 else y
        ana_values = np.concatenate([ana_values, ana],axis=0) if len(ana_values) > 0 else ana
        while len(x_values)>filesize:
            np.save(xy_test_path.joinpath(f"x_{index:03d}"),x_values[:filesize])
            np.save(xy_test_path.joinpath(f"y_{index:03d}"),y_values[:filesize])
            np.save(xy_test_path.joinpath(f"ana_{index:03d}"),ana_values[:filesize])
            index+=1
            x_values = np.delete(x_values, np.s_[0:filesize], axis = 0)
            y_values = np.delete(y_values, np.s_[0:filesize], axis = 0)
            ana_values = np.delete(ana_values, np.s_[0:filesize], axis = 0)
            print(np.shape(x_values))


#Run sort_elems once to sort data for all years (f"L2_1[4-8]*.npy" can be changed for selecting years).
#Aftrewards run sort_arrays for every single year.

sort_elems()
sort_arrays(year="2014")
sort_arrays(year="2015")
sort_arrays(year="2016")
sort_arrays(year="2017")
sort_arrays(year="2018")
sort_arrays(year="2019")
