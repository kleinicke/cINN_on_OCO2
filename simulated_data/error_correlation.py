import config as c
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import data.data_helpers as data_helpers
import data.prepare_data as prepare_data
import evaluate
from evaluate import compute_calibration,sample_posterior
import time
import torch

#compute_calibration = evaluate.compute_calibration
#sample_posterior = evaluate.sample_posterior
x_all, y_all, ana_all = data_helpers.concatenate_test_set(c.evaluation_samples)

def show_error_correlations(x=x_all,y=y_all,load=False):

    #plot error co2 against: xco2, "albedo_o2","albedo_sco2","albedo_wco2", "tcwv" 4
    #and: "year","xco2_apriori","altitude","psurf","t700","longitude","latitude" 7
    #"snr_wco2","snr_sco2","snr_o2a","aod_bc","aod_dust","aod_ice","aod_oc","aod_seasalt","aod_sulfate","aod_total","aod_water"

    params=prepare_data.x_to_params(x)
    spectra=prepare_data.y_to_spectra(y)

    post = sample_posterior(y)
    diff = torch.mean(torch.FloatTensor(post), dim = 1)-x
    #post_params =prepare_data.x_to_params(post)
    _,_,uncert_intervals,_, post_params = compute_calibration(x,y,load=load)
    uncert_error_co2=uncert_intervals[68,:,c.co2_pos]/2
    diff = np.mean(post_params, axis = 1) - params
    diff_co2 = diff [:, 0]
    error_name = ['error_correlations', 'estimated_error_correlations']
    for l,spectra in enumerate([spectra,np.array(y)]):
        if l==1:
            params = np.array(x)
        for k, diff in enumerate([diff_co2, uncert_error_co2]):
        
            plt.figure(error_name[k]+f'_{l}',figsize=(20,15))
            plt.title(error_name[k]+f'_{l}')    
            print(diff.shape)
            horizontal_figures = 4
            vertical_figures = 6
            diff = np.clip(diff,-4,4)
            for i in range(horizontal_figures):
                ax = plt.subplot(horizontal_figures, vertical_figures, vertical_figures*i+1)


                bins=np.linspace(np.min(diff),np.max(diff),100)

                plt.hist(diff, bins=bins, histtype='step',color="lightskyblue", orientation="horizontal")


                
                if i>0:
                    #ax.axis('off')
                    ax.set_xticks([])
                plt.ylabel(f"error of prediction in ppm")
            """
            for i in range(horizontal_figures):
                for j in range(vertical_figures-1):
                    ax = plt.subplot(horizontal_figures, vertical_figures, vertical_figures*i+j+2)
                    if i == 0:
                        ax.xaxis.tick_top()
                        #plt.xlabel(c.param_names[i])
                        ax.set_xlabel(c.params_names[j])
                        ax.xaxis.set_label_position('top')
                        #plt.scatter(params[:, j],diff[:, 0],s=0.5,alpha=0.2,color="green")
                        plt.scatter(params[:, j],diff,s=1,alpha=0.3,color="blue")
                        #plt.ylabel(f"{prepare_data.params_names[j]}")
                    if i == 1:
                        plt.scatter(spectra[:, -j-1],diff,s=1,alpha=0.3,color="blue")
                        #plt.ylabel(f"{prepare_data.spectra_names[-j]}")
                        ax.set_xlabel(c.params_names[-j-1])
                        ax.xaxis.set_label_position('top')
                    if i == 2:
                        plt.scatter(spectra[:, -j-5-1],diff,s=1,alpha=0.3,color="blue")
                        ax.set_xlabel(f"{c.params_names[-j-5-1]}")
                    if i == 3:
                        plt.scatter(spectra[:, -j-10-1],diff,s=1,alpha=0.3,color="blue")
                        ax.set_xlabel(f"{c.params_names[-j-10-1]}")
            """
    

if __name__ == "__main__":
    show_error_correlations(load=False)
    plt.show()