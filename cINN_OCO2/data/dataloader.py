import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

from pathlib import Path

#required, since it's depending from which folder you call this script.
try:
    import data.prepare_data as prepare_data
    import data.data_config as dc
except:
    import prepare_data as prepare_data
    import data_config as dc




class loadOCODataset(Dataset):
    """OCO-2 CO2 Spectra dataset."""

    def __init__(self, root_dir=Path(__file__).parent.joinpath("files/"),year=[2014,2015], fast = True, noise=True, select = False,analyze = True, number = dc.preprocessing_mode, mask = []):
        """Dataset that can load normalized or unnormalized spectral data and corresponding paramerter.
        
        Keyword Arguments:
            root_dir {Path} -- Path to dataset (default: {Path(__file__).parent.joinpath("files/")})
            year {list} -- years to load (default: {[2014,2015]})
            fast {bool} -- Should load data to memory? (default: {True})
            noise {bool} -- apply noise on data? (use this for training but not for testdata) (default: {True})
            select {bool} -- select a subset of data, to get f.e. less testdata for faster testing (default: {False})
            analyze {bool} -- return more data, that can be used to analyse (default: {True})
            number {[type]} -- is loaded data already normalized?(then its 1) (default: {dc.preprocessing_mode})
        """
        if noise:
            print(f"\nLoads data of year {year} with noise.")
        else:
            print(f"\nLoads data of year {year} w.o. noise.")
        self.root_dir = root_dir
        self.analyze = analyze
        self.noise = noise
        ###np.random.seed(0)
        self.number = number
        x_path = []
        y_path = []
        ana_path = []
        #loads all the datafile paths in 3 lists
        if self.number == 1:
            #default
            name_add = "xy"
            x_name = "x"
            y_name = "y"
        elif self.number == 2:
            name_add = "spectraparams"
            x_name = "params"
            y_name = "spectra"
        for y in year:
            x_path += sorted(root_dir.glob(f"4_{name_add}_{y}{dc.short_str}/{x_name}_*.npy"))
            y_path += sorted(root_dir.glob(f"4_{name_add}_{y}{dc.short_str}/{y_name}_*.npy"))
            ana_path += sorted(root_dir.glob(f"4_{name_add}_{y}{dc.short_str}/ana_*.npy"))               

        x_path=x_path
        y_path=y_path
        ana_path=ana_path
        assert len(x_path)==len(y_path),(len(x_path),len(y_path),x_path)
        assert len(x_path)==len(ana_path),(len(x_path),len(ana_path),x_path,ana_path)

        #loads the data from all paths
        self.mask = mask
        #mask = []
        x_list = []
        y_list = []
        ana_list = []
        for j in range(len(x_path[:])):
            #check that numbers match
            number = str(x_path[j]).split('/')[-1:][0].strip()[2:5]
            number2 = str(y_path[j]).split('/')[-1:][0].strip()[2:5]
            number3 = str(ana_path[j]).split('/')[-1:][0].strip()[4:7]
            if self.number == 2:
                number = str(x_path[j]).split('/')[-1:][0].strip()[7:10]
                number2 = str(y_path[j]).split('/')[-1:][0].strip()[8:11]
                number3 = str(ana_path[j]).split('/')[-1:][0].strip()[4:7]
            assert number == number2,f"error: {number,number2}, {x_path[j]}"
            assert number == number3,f"error: {number,number3}"
            #print name without newline
            print(f"{j:3d}",number, end = ', ')
            if j%10==9:
                #newline
                print("")
            
            if dc.short_ds and fast:
                #load data into ram
                x = np.load(str(x_path[j]))
                y = np.load(str(y_path[j]))
                ana = np.load(str(ana_path[j]))
                if self.number == 2:
                    #print("computes log")
                    np.log(y[:,:3*prepare_data.spec_length],out = y[:,:3*prepare_data.spec_length])
                    y = np.dot(y - prepare_data.mu_y, prepare_data.w_y)
                    #assert 0
            else:
                #loads data but keeps in on the harddrive.
                #Usefull if data is bigger than ram size.
                x = np.load(str(x_path[j]),mmap_mode = 'c')
                y = np.load(str(y_path[j]),mmap_mode = 'c')
                ana = np.load(str(ana_path[j]),mmap_mode = 'c')
            if select:
                #reduces the amount of data. Usefull f.e. to reduce testsetsize for faster testing
                if len(mask) == 0:
                    print("creates mask")
                    mask = np.full(len(x), False)
                    mask[:int(len(x)/20)] = True
                    np.random.shuffle(mask)
                x=x[mask]
                y=y[mask]
                ana = ana[mask]

            #store all the loaded data in lists
            x_list.append(x)
            y_list.append(y)
            ana_list.append(ana)
        self.y_list = y_list
        self.x_list = x_list
        self.ana_list = ana_list
        self.filesize = len(x_list[0])
        print(f"loads {len(x_list)} files of size {self.filesize}.")
        print(f"Shape of files is x:{x_list[0][0].shape}, y:{y_list[0][0].shape}, ana:{ana_list[0][0].shape}")

        i_snr = prepare_data.spectra_names.index("snr_wco2")
        self.i_rel_snr = i_snr - len(prepare_data.spectra_names)
        assert self.i_rel_snr == -18,(self.i_rel_snr,i_snr,len(prepare_data.spectra_names),y[-20:])

    def __len__(self):
        """Return lenght of dataset
        """
        return len(self.x_list)*self.filesize

    def __getitem__(self, idx):
        """returns single data to dataloader
        """

        #calculate in which file demanded sample is
        file_number = int(idx/self.filesize)
        index = idx%self.filesize

        #loads sample
        x = self.x_list[file_number][index] 
        y = self.y_list[file_number][index] 
        ana = self.ana_list[file_number][index]

        if self.noise and self.number == 2 and False:
            print("Not Good")
            #i_snr = prepare_data.spectra_names.index("snr_wco2")
            #i_rel_snr = i_snr - len(prepare_data.spectra_names)
            #assert i_rel_snr == -18,(i_rel_snr,i_snr,len(prepare_data.spectra_names),y[-20:])
            noise = np.random.normal(loc = 0, scale = 1,size=prepare_data.spec_length)
            #y[:prepare_data.spec_length] += (noise*y[prepare_data.spec_length:2*prepare_data.spec_length])/y[self.i_rel_snr]
            #print(noise,y[prepare_data.spec_length:2*prepare_data.spec_length],y[self.i_rel_snr],(noise*y[prepare_data.spec_length:2*prepare_data.spec_length])/y[self.i_rel_snr])
            #assert 0
            noise = np.random.normal(loc = 0, scale = 1,size=prepare_data.spec_length)
            #y[prepare_data.spec_length:2*prepare_data.spec_length] += noise*y[prepare_data.spec_length:2*prepare_data.spec_length]/y[self.i_rel_snr+1]
            noise = np.random.normal(loc = 0, scale = 1,size=prepare_data.spec_length)
            #y[2*prepare_data.spec_length:3*prepare_data.spec_length] += noise*y[2*prepare_data.spec_length:3*prepare_data.spec_length]/y[self.i_rel_snr+2]
        #if it wasn't normalized yet
        if self.number == 2:
            x = x[prepare_data.params_mask]
            x = prepare_data.params_to_x(x)
            #y = prepare_data.dcopy(y)
            #np.log(y[:3*prepare_data.spec_length],out = y[:3*prepare_data.spec_length])
            #y = prepare_data.spectra_to_y((y[None,:]))[0]#prepare_data.dcopy
            #y = y - prepare_data.mu_y
            #print(np.max(y))
            
            assert not np.isnan(np.sum(x))
            assert not np.isnan(np.sum(y))   

        #apply noise if wanted
        if self.noise:
            rand = np.random.normal(loc = 1, scale = dc.train_noise_sigma,size=x.size)
            x = x*rand
            rand = np.random.normal(loc = 1, scale = dc.train_noise_sigma,size=y.size)
            y=y*rand
            if dc.additional_noise:
                x[0] += np.random.normal(loc = 0, scale = ana[5]*prepare_data.w_x[0,0])#*0.2
                x[4] += np.random.normal(loc = 0, scale = ana[3]*prepare_data.w_x[4,4])#*0.2
            

        #transform to torch data
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        ana = torch.Tensor(ana)
       

        if self.analyze:
            return (x,y,ana) 
        else:
            return x,y

data_size = 1000
perm = np.random.permutation(data_size)

testMask = np.ones(data_size,dtype=bool)
testMask[100:] = False
testMask = testMask[perm]
trainMask = np.ones(data_size,dtype=bool)
trainMask[:100] = False
trainMask = trainMask[perm]

#print(testMask)
#print(trainMask)

#gets train and testset, to be used by dataloader
ana_train_set = loadOCODataset(analyze=True, year = dc.train_years, fast = True, mask = trainMask, select=False)
#ana_train_set = loadOCODataset(root_dir = Path(__file__).parent.joinpath("files/train"), noise=False)
#ana_test_set = loadOCODataset(root_dir = Path(__file__).parent.joinpath("files/test"), analyze=True, noise=False)
#ana_test_set = loadOCODataset(year = dc.test_years, analyze=True, select=True, noise=False)#

#when it's already a short dataset, the complete set can be used for testing 
if dc.short_ds:
    ana_test_set = loadOCODataset(year = dc.test_years, analyze=True, noise=False, mask = testMask, select=False)
else:
    ana_test_set = loadOCODataset(year = dc.test_years, analyze=True, select=True, noise=False, mask = testMask)#                                                                                                                                                                                                                                                                                                                                                                                                                            
#ana_test_set = loadOCODataset(analyze=True, year = dc.test_years, fast = True, mask = trainMask, select=False, noise=False)

#get parameter out of the loaded data
print(f"trainset size: {ana_train_set.__len__()} testset size: {ana_test_set.__len__()}")
trainset_size = ana_train_set.__len__()
testset_size = ana_test_set.__len__()
parameter_size,spectrum_size = int(np.sum(prepare_data.params_mask)),len(ana_train_set.y_list[0][0])#len(train_set.x_list[0][0])

print("sizes of parameter and spectrum are ",parameter_size,spectrum_size)


def get_ana_loaders(batch_size, seed=0):
    """Create dataloader for test and train ana_set.

    Arguments:
        batch_size {int} -- Batch size of loader

    Keyword Arguments:
        seed {int} -- torch seed of loader (default: {0})

    Returns:
        [test_ana_loader, train_ana_loader] -- test and train dataloader
    """
    ###torch.manual_seed(seed)
    test_ana_loader = DataLoader(ana_test_set,
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 8)

    train_ana_loader = DataLoader(ana_train_set,
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers= 8)
    return test_ana_loader, train_ana_loader




#create dataset for visualization for each single year.
try:
    import data.data_helpers as data_helpers
except:
    import data_helpers as data_helpers
loader = []
year_sets = {}

for _, year in enumerate(dc.viz_years):
    sets = loadOCODataset(year = [year], analyze=True, noise=False, fast = False, number=1)
    #sets = loadOCODataset(analyze=True, year = [year], fast = False, mask = trainMask, select=False)
    loadset = DataLoader(sets,
        batch_size=512, shuffle=True, drop_last=True, num_workers = 8)
    loader.append(loadset)
    x,y,_ = data_helpers.concatenate_set(loadset,dc.viz_size)
    year_sets[year] = (x,y)

if dc.sum_of_years:
    #to analyze difference between test set and single years
    year = "train_set_tester"
    dc.viz_years += [year]
    loadset = DataLoader(ana_train_set,
            batch_size=512, shuffle=True, drop_last=True, num_workers = 8)
    loader.append(loadset)
    x,y,_ = data_helpers.concatenate_set(loadset,dc.viz_size)
    year_sets[year] = (x,y)
    

