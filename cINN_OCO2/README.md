# conditional Invertible Neural Network applied on OCO-2 Spectra

This instruction is done for a example, where the data is already available.
When no data is available, generate it using the master's thesis or the instruction

```https://github.com/kleinicke/MatchL1L2``` 

to download and match the data.
Afterward follow the `data/files/README.md` instruction to create a dataset that is ready for the dataloader.



## Installation

`python3.6` is required. (tested with 3.7 only)

First install the Freia Framework  

```https://github.com/VLL-HD/FrEIA```

For evaluation during training we use `tensorflow/tensorboard`. 
When this is not installed it has to be installed or the code as to be deleted.  

For evaluation after training `geopandas` is required.

## Preparation

Afterward make sure the data is correctly stored in the `data/files folder`.
Within that folder contains the folders `1_sp_samples` and`4_spectraparams_2014_short` to `4_spectraparams_2019_short` with data inside and the folders `2_normed` and `4_xy_2014_short` to `4_xy_2019_short` without data inside.

In this main folder the folders `tmp/log_dir`, `output` and `images` have to be created.

The script was used on a Linux computer with 32 GB of memory. It is used, for loading the dataset and a year wise dataset into memory. When only 16 GB are available the second loading of the dataset should be deleted.

Before running the code, the number of maximal loadable files has to be increased using

``` ulimit -n 2048 ```

To see the training process, tensorboard has to be called: `tensorboard --logdir=tmp/log_dir`

## data folder

In the datafolder are some important files.

the `data/files` folder should be linked to the real data folder using `ln -s <Path>`.
`data/prepare_data.py` takes the raw data, and can normalize it if required.
Important variables for this are set in `data/data_config.py`.
One of the variables are `create_new_norm`. When this is set, the raw data is normalized again.
The normalized data is used by `data/dataloader.py`. It creates a dataloader for the test and train set and for every single year. 
The years that belong to the test and train set are also defined in `data/data_config.py`.
`plot_data.py` can be used to plot the raw data to find interesting relations or errors.
 
## Training

For training run the script `train.py`.

All parameter imporant for the training are set in `config.py`.

The feature network is defined in `nets.py` and the cINN in `model.py`.

After training the evaluation script
`nice_eval.py` is automatically called.

`error_correlation.py`, `help_train.py` and `plot_helpers.py` support the training and evaluation process.

## Evaluation
To evaluation are two scripts available. `nice_eval.py` that got all the current figures and `old_evaluate.py` that got some more figures, but they are not so far optimized.

For evaluating the data use the script `data/plot_data.py`. 
Therefore `seaborn` has to be installed using pip.
