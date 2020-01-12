import numpy as np
import glob
import re
import geopandas
import torch
import seaborn as sns


try:
    import data.data_config as dc
    dc.viz_size = 10000
    import data.prepare_data as prepare_data
    import data.dataloader as dataloader
    import data.data_helpers as data_helpers
    #from data.prepare_data import wavelenth_y,output_y,output_x,params_names,output_params,normizing_factors
    #from data.prepare_data import mu_x,mu_y,w_x,w_y, all_params_names, external_params
    print("loaded data.prepare_data")
except:
    import data_config as dc
    dc.viz_size = 10000
    import prepare_data
    import dataloader
    import data_helpers
    #from prepare_data import wavelenth_y,output_y,output_x,params_names,output_params,normizing_factors
    #from prepare_data import mu_x,mu_y,w_x,w_y, all_params_names, external_params
    print("loaded prepare_data")

#import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

co2_pos = 0#len(prepare_data.output_params[0])-4
params_in_spectrum = len(prepare_data.spectra_names)-3

"""Plotting of input data histogramms and qualitative checking of single spectra and parameter
"""
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 
n_parameters=len(prepare_data.params_names)#29#30#9#parameter_size
#n_observ=5
batch_size = 128
#test_loader, train_loader = dataloader.get_loaders(batch_size)


x_all, y_all, ana_all = data_helpers.concatenate_train_set(1000)#concatenate_test_set(1000)#400)#20)#400)
print(torch.max(y_all), torch.min(y_all))
print(torch.max(y_all[:,-params_in_spectrum:], dim = 0), torch.min(y_all[:,-params_in_spectrum:], dim = 0))
print(torch.max(y_all[:,:prepare_data.spec_length]), torch.min(y_all[:,:prepare_data.spec_length]))
print(torch.max(y_all[:,prepare_data.spec_length:2*prepare_data.spec_length]), torch.min(y_all[:,prepare_data.spec_length:2*prepare_data.spec_length]))
print(torch.max(y_all[:,2*prepare_data.spec_length:3*prepare_data.spec_length]), torch.min(y_all[:,2*prepare_data.spec_length:3*prepare_data.spec_length]))

x_all, y_all, ana_all = data_helpers.concatenate_test_set(1000)#concatenate_test_set(1000)#400)#20)#400)
print(torch.max(y_all), torch.min(y_all))
#assert 0

#x=prepare_data.x
#params=prepare_data.params
#spectra = prepare_data.spectra
#params_in_spectrum = prepare_data.number_external_params #5 or 18
#params_names = prepare_data.params_names#list(prepare_data.namedict.keys())
#spec_length = prepare_data.spec_length

def show_param_hists(parameters,n_parameters=n_parameters,offset=0,name="", elem_names = prepare_data.params_names):
        plt.figure(f'{name} parameter histograms')
        plt.title('parameter histograms')
        for i in range(n_parameters):
            plt.subplot(n_parameters/4+1, 4, i+1)
            print(parameters[:, i+offset])
            plt.hist(parameters[:, i+offset], bins=50, histtype='step')
            plt.xlabel(f"{elem_names[i]}")
        plt.tight_layout()

def show_obs_hists(observations):
    plt.figure()
    for i in range(n_observ):
        plt.subplot(3, 3, i+1)
        try:
            plt.hist(observations[:, i], bins=50, histtype='step')
        except ValueError:
            continue

    plt.tight_layout()


def show_spectra(number = 3):
    spec_length = prepare_data.spec_length
    spectra = prepare_data.y_to_spectra(y_all)[:,:-prepare_data.params_in_spectrum]
    logspectra = np.log(spectra)#spectra                
    #logspectra = spectra                

    for i in [0,5,10,15,18,20,24,28]:
        i=i*10
        print (len(spectra),i, y_all.shape, spectra.shape)
        if len(spectra) > i:
            plt.figure(f"spectra {i}",figsize=(15,15))
            
            for j in range(number):
                ax = plt.subplot(number,1,j+1)
                #logspectra = np.log(spectra)                
                plt.plot(range(spec_length),(logspectra[i,spec_length*j:spec_length*(j+1)]))#wavelenth_y[0,:-params_in_spectrum])
                ax.set_xlabel(r"Wavelength in ${\mu}m$")
                ax.set_ylabel(r"Radiance in $log_{10}\;\frac{Ph}{sec\:m^{2}sr\; {\mu}m}$")
                if j == 0 :
                    ax.set_title(r"Strong CO$_2$ Band")
                elif j == 1:
                    ax.set_title(r"Weak CO$_2$ Band")
                else:
                    ax.set_title(r"O$_2$ Band")

                print("\n")
                print("spectrum ",i,j)
                #print(wavelenth_y[0,:-params_in_spectrum], np.shape(wavelenth_y[0,:-params_in_spectrum]))
                print(logspectra[i,spec_length*j:spec_length*(j+1)], np.shape(logspectra[0,spec_length*j:spec_length*(j+1)]))
                #print(spectra[i,:])
                #print(params[i,:])

def show_param_corr(parameters):
    """
    Shows the correlation of certain parameter
    """
    plt.figure()
    cii = np.corrcoef(parameters, rowvar=False)
    for i, C in enumerate([cii, np.log10(np.abs(cii)), np.log10(1-cii)]):
        plt.subplot(3,1,i+1)
        plt.imshow(C)
        plt.colorbar()
    plt.tight_layout()

def show_obs_corr(observations):
    """shows correlation of certain spectra

    """

    cii_obs = np.corrcoef(observations, rowvar=False)
    for i, C in enumerate([cii_obs, np.log10(np.abs(cii_obs)), np.log10(1-cii_obs+1e-7)]):
        plt.figure()
        plt.imshow(C)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(F'correlations_{i}.png', dpi=450)

def print_correlation():
    corr=np.zeros((8,20))
    for i in range(8):
        corr[i]=x[:20,i+1]/x[:20,0]
    print(corr)


def show_elementcorrelations(parameters=x_all,n_parameters=n_parameters,name="",params_names=prepare_data.params_names):
    """Creates nice correlation plot of the given input parameters
    It's useful to see, if certain parameter are independent of each other of if using less input parameter doesn't change the impact on the problem.
    
    Keyword Arguments:
        parameters {[type]} -- [the array of parameter] (default: {orig_parameters})
        n_parameters {[int]} -- [number of compared parameter] (default: {n_parameters})
        offset {[int]} -- [number of first compared parameter in parameters] (default: {offset})
        name {str} -- [name, to show in plot] (default: {""})
        params_names {[str]} -- [names of parameters] (default: {params_names})
    """

    plt.figure('correlations'+name,figsize=(20,15))
    plt.title('correlations') #isn't visible anyway
    plt.tight_layout()

    #Used to ignore certain input data that is an outlier. Only usefull for nicer plotting
    use_mask=False
    if use_mask:
        CH4_mask=np.less(parameters[:,-1],1.67)
        CO2_low_mask=np.less(parameters[:,co2_pos],377)
        CO2_high_mask=np.less(389,parameters[:,co2_pos])
        CO2_mask=np.logical_or(CO2_low_mask,CO2_high_mask)
        Pressure_mask=np.less(parameters[:,0],850)
        mask=CH4_mask


    #print(np.shape(parameters))

    for i in range(n_parameters-1,-1,-1):
        ax=plt.subplot(n_parameters+1, n_parameters, i+1)
        bins=np.linspace(np.min(parameters[:, i]),np.max(parameters[:, i]),100)
        plt.hist(parameters[:, i], bins=bins, histtype='step')
        if use_mask:
            plt.hist(parameters[:, i][np.logical_not(mask)], bins=bins, histtype='step',color="green")
            plt.hist(parameters[:, i][mask], bins=bins, histtype='step',color="orange")
        ax.xaxis.tick_top()
        #if i==co2_pos:
        #    ax.set_xlim(375,390)
        if i>0:
            ax.set_yticks([])
        else:
            plt.ylabel(f"Prior dist")

    for i in range(n_parameters):
        for j in range(n_parameters):
            ax=plt.subplot(n_parameters+1, n_parameters, i+1+n_parameters*(j+1))
            plt.subplots_adjust(hspace = .001, wspace=.001)
            if use_mask:
                plt.scatter(parameters[:, i][np.logical_not(mask)],parameters[:, j][np.logical_not(mask)],s=0.1,alpha=0.1)
                plt.scatter(parameters[:, i][mask],parameters[:, j][mask],s=0.1,alpha=0.1,color="orange")
            else:
                plt.scatter(parameters[:, i],parameters[:, j],s=0.1,alpha=0.1)
     
            #if i==co2_pos:
            #    ax.set_xlim(375,390)
            #if j==co2_pos:
            #    ax.set_ylim(375,390)
            if j<n_parameters-1:
                ax.set_xticks([])
            if i>0:
                ax.set_yticks([])
            else:
                if len(params_names[j])>7:
                    plt.ylabel(f"{params_names[j][:4]}..{params_names[j][-2:]}")
                else:
                    plt.ylabel(f"{params_names[j]}")
        plt.xlabel(params_names[i])

def plot_quality_map(value,name="co2"):
    """AKA world map

    
    Arguments:
        uncert_intervals {[type]} -- [description]
    
    Keyword Arguments:
        name {str} -- [description] (default: {"Satelite positon"})
        title {str} -- [description] (default: {"Width of uncertanty, depending on position"})
    """


    
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    fig = plt.figure(f"{name}_map",figsize=(20,10))
    plt.tight_layout()
    #plt.title(title)
    ax = plt.subplot(1,1,1)
    #fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(20,15))
    fig.tight_layout()
    #fig.set_title(title)
    #print(np.shape(external_results), np.shape((error)), np.shape(month_masks))
        #ax = plt.subplot(2, 2, 1)
    ax.set_aspect('equal')
    ax.set_title(f"{name}")
    world.plot(ax=ax, color='lightblue', edgecolor='black')
    pos = prepare_data.y_to_spectra(y_all)[:,-2]
    #prepare_data.position
    print(pos)
    plt.scatter(pos[:,0],pos[:,1],c=value,s=20)#,cmap=plt.get_cmap("jet"))#"plasma"
        #plt.colorbar(im)
    #plt.legend(title="in ppm")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")   
    
    plt.colorbar()


def show_co2_change():

    #def hist(x):
    #    results = []
    #    h, b = np.histogram(x[:, 0], bins=100, density=True)#range=(-2,2),
    #    h /= np.max(h)
    #    results = [b[:-1],h]
    #    return results


    torch.manual_seed(153)
    plt.figure(f"CO2 concentration",figsize=(10,7), dpi=100)
    ax = plt.gca()
    means = []
    for _, year in enumerate(dc.viz_years):
        x,y = dataloader.year_sets[year] 
        
    #for _, year in enumerate([2014, 2015, 2016, 2017, 2018]):

        #sets = dataloader.loadOCODataset(year = [year], analyze=True, noise=False)
        #loadset = dataloader.DataLoader(sets,
        #    batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 1)
        #x,_,_ = data_helpers.concatenate_set(loadset,10000)
        params = prepare_data.x_to_params(x)


        #params_hist = hist(params)
        #print(params.min(axis=0))
        #print(params.max(axis=0))
        #plt.subplot(5, 5/5+1, j +1)
        #plt.step(*(params_hist), where='post', label = year) 
        color = next(ax._get_lines.prop_cycler)['color']
        #print(np.shape(params_hist))
        sns.set_style("whitegrid")
        sns.kdeplot(params[:,0], bw = 0.2, label = year, color = color)
        mean = np.mean(params[:,0])
        means.append(mean)
        plt.plot([mean, mean], [0,0.25], color=color)
        
    title = "The means of the years were:"
    for mean in means:
        title += f" {mean:.2f},"
    differences = ""
    prev_mean = means[0]
    for mean in means[1:]:
        differences += f"{mean-prev_mean:.2f}, "
        prev_mean = mean
    plt.title(title[:-2])
    plt.legend()
    #plt.tight_layout()
    plt.xlabel(f"CO2 in ppm, differences {differences[:-2]}")
    
    
    
    plt.figure(f"Mean corrected CO2 distribution",figsize=(10,7), dpi=100)
    ax = plt.gca()
    means = []
    for _, year in enumerate(dc.viz_years):
        x,y = dataloader.year_sets[year] 
        
    #for _, year in enumerate([2014, 2015, 2016, 2017, 2018]):

        #sets = dataloader.loadOCODataset(year = [year], analyze=True, noise=False)
        #loadset = dataloader.DataLoader(sets,
        #    batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 1)
        #x,_,_ = data_helpers.concatenate_set(loadset,10000)
        params = prepare_data.x_to_params(x)


        #params_hist = hist(params)
        #print(params.min(axis=0))
        #print(params.max(axis=0))
        #plt.subplot(5, 5/5+1, j +1)
        #plt.step(*(params_hist), where='post', label = year) 
        color = next(ax._get_lines.prop_cycler)['color']
        #print(np.shape(params_hist))
        sns.set_style("whitegrid")
        mean = np.mean(params[:,0])
        sns.kdeplot(params[:,0]-mean, bw = 0.2, label = year, color = color)
        means.append(mean)
        #plt.plot([mean, mean], [0,0.25], color=color)
        
    title = "Mean corrected distribution:"
    for mean in means:
        title += f" {mean:.2f},"
    differences = ""
    prev_mean = means[0]
    for mean in means[1:]:
        differences += f"{mean-prev_mean:.2f}, "
        prev_mean = mean
    plt.title(title[:-2])
    plt.legend()
    #plt.tight_layout()
    plt.xlabel(f"Distribution shapes, 2014 and 2019 not complete years")

def show_month_map():
    month_set = {}

    for i in range(1,10):
        x,y = dataloader.year_sets[2019]
    

    


def plot():
    #for i in range(len(params[0])):
    #    plot_quality_map(params[:,i],name = params_names[i])
    show_co2_change()
    params = prepare_data.x_to_params(x_all)
    specs = prepare_data.y_to_spectra(y_all)
    #plot_quality_map(params[:,co2_pos])
    show_param_hists(x_all,name="whightened")
    show_param_hists(params,name="orig")
    last_specs = specs[:,-20:]
    last_specs[:,6] /= 12
    show_param_hists(last_specs,name="spec-20", n_parameters = 20, elem_names = prepare_data.spectra_names[-20:])
    show_param_hists(y_all[:,-20:],name="y-20", n_parameters = 20, elem_names = prepare_data.spectra_names[-20:])
    #show_elementcorrelations()
    #spectra_positions = [0,spec_length,2*spec_length]
    #for i in range(params_in_spectrum):
    #    spectra_positions.append(3*spec_length+i)

    #show_elementcorrelations(spectra[:,spectra_positions],len(prepare_data.spectra_names),"external params",params_names=prepare_data.spectra_names[:])
    #show_one_input_data()
    show_spectra()
    #show_param_hists(new_params,n_parameters=5,name="0")


if __name__ == "__main__":
    plot()
    show_co2_change()

    plt.show(block=False)
    input("Press enter to end!")