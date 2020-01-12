# Files Generation

## Link existing files

Replace this folder by linking this folder to the correct data using `ln -s <Path>`

## Preparation

To generate the files, first follow

```https://github.com/kleinicke/MatchL1L2```

Afterward run the script `create_folders.sh`. This will create the folders that are required to further generate the dataset.  
Copy the resulting files `npy` files from the other git folder from `generate/npy` to `0_filtered_L1L2`.

## Create data for dataset

The next step is to run `postprocess.py`. It's function `sort_elems()` creates samples in the `1_sp_samples/` folder which can be used to normalize the files.
It will filter the files and save them in `1_spectraparams`. The filtering currently will reduce the size to about 14 GB.  
The files will differ in the amount of samples. The `sort_arrays()` function will chunk the file in peaces of 1000 samples and 
stores them in the folders `4_spectraparams_2014_short` to `4_spectraparams_2019_short`. Here they can be used by `prepare_data.py` and `dataloader.py`.
