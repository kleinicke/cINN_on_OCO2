import numpy as np
import glob
import re
"""Reads Data from simulated files. 
Resulting files can be transferred to other server 
for further preprocessing for the iNN.
"""

#Start with: [13:26],[26:39],[10:13],[103],[146,147],[131]

#REMOTEC-OUTPUT_X_APR_STATE X_TYPE0001 , X_TYPE0006 [146,147]

#REMOTEC-OUTPUT_X_STATE ALB_WIN01_ORDER00 [131]
epsilon=1e-10


elements=[]
for i in range(10,39):
    elements.append(i)
for i in [103,131,146,147]:
    elements.append(i)
#[range(10,13),range(13,26),range(26,39),103]
print(elements)
readSynth=True


#TODO: Check if spect is matching
#TODO add external params to spect data. 
external_elements=[]
for i in range(10,13):
    external_elements.append(i)

def save_spect(lists=[5,6,7,10,12],split=""):
    """Reads the spectra from given files and saves them.

    Keyword Arguments:
        lists {list} -- [description] (default: {[5,6,7,10,12]})
        split {str} -- [description] (default: {""})

    Returns:
        np_array -- All spectra
    """

    longspect=[]
    #lists=[5,6]
    lists_name=""
    for i in lists:
        file = open(f"LST/Split{split}/l{i}.lst", "r") 

        print(f"loads file: LST/Split{split}/l{i}.lst")
        for line in file: 

            longspect.append(np.genfromtxt("SYNTH_SPECTRA/L1B_"+line.rstrip(), skip_header=15))
        print("longspect",np.shape(longspect))
        lists_name+="_"+str(i)

    np.save(f"CONTRL_OUT/Files/raw_spectrum{lists_name}.npy",longspect)
    np.save(f"CONTRL_OUT/Files/raw_spectrum.npy",longspect)
    longspect=np.asarray(longspect)[:,:,1]
    return longspect

def save_params(lists=[5,6,7,10,12],split=""):
    """Reads Parameter from retrieved files and saves them.

    Keyword Arguments:
        lists {list} -- [description] (default: {[5,6,7,10,12]})
        split {str} -- [description] (default: {""})
    """

    longparam = np.array([])
    lists_name=""
    for i in lists:
        try :
            param_i = np.genfromtxt(f"CONTRL_OUT/short_id{i:04d}.out")
            longparam= np.vstack([longparam, param_i]) if longparam.size else param_i
            lists_name+="_"+str(i)
        except:
            param_i=[0]
            print(f"getting parameters from short_id{i:04d}.out failed.")
        num_lines = sum(1 for line in open(f'LST/Split{split}/l{i}.lst'))
        #print(np.shape(param_i)[0])
        if num_lines > np.shape(param_i)[0]:
            print(f"{bcolors.FAIL}param_{i} {np.shape(param_i)}, lines {num_lines}{bcolors.ENDC}")
        else:
            print(f"{bcolors.OKGREEN}param_{i} {np.shape(param_i)}, lines {num_lines}{bcolors.ENDC}")

    np.save(f"CONTRL_OUT/Files/raw_parameters{lists_name}.npy",longparam)
    np.save(f"CONTRL_OUT/Files/raw_parameters.npy",longparam)
    print(f"Saved longparams {np.shape(longparam)} for {lists_name}")


lists=[50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100]
good_lists=[50,150,200,250,300,350,550,600,650,700,850,1000,1050,1100]
save_params(good_lists,"2")
save_spect(good_lists,"2")
