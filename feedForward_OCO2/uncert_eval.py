import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import scipy
from scipy.stats import norm
import time

import data.prepare_data as prepare_data
import data.dataloader as dataloader
import data.data_helpers as data_helpers
import config as c

import help_train as ht
import plot_helpers
import feed_forward_net as feature_net


model = feature_net.model
feature_net.load(c.feature_net_file)
model.eval()
#model = c.model
model.to(c.device)

#model.load(c.filename)
model.eval()
#print('Trainable parameters:')
#print(sum([p.numel() for p in model.trainable_parameters]))

#Variables:
# how many different posteriors to show:
n_plots = 20

# how many dimensions of x to use:

print(f"Evaluates with n_plots={n_plots}")

x_all, y_all, ana_all = data_helpers.concatenate_test_set(c.evaluation_samples)

position = prepare_data.y_to_spectra(y_all)[:,-2:]
uncert_index = prepare_data.ana_names.index("xco2_uncertainty")

def compute_uncert(x,y, load=False,save_name="test"):
    params=prepare_data.x_to_params(x)


    print("Starts to sample posteriors for calibration errors")

    t = time.perf_counter()
    predictions = model.fn_func(y_all.to(c.device)).detach().cpu().numpy()
    x_pred = predictions[:,::2]
    uncert_x_pred = np.sqrt(np.exp(predictions[:,1::2]))
    params_pred=prepare_data.x_to_params(x_pred)
    uncert_params_pred=prepare_data.x_to_params(uncert_x_pred,no_mu=True)
    diff = params_pred - params
    print("mean denorm",np.mean(diff, axis = 0))
    print("absmean denorm",np.mean(np.abs(diff), axis = 0))   
    print(f"sample and x to params time: {time.perf_counter() - t:.2f}s")
    print(f"Total time to compute inlier matrix: {time.perf_counter() - t:.2f}s")
    calibration = diff/uncert_params_pred
    print("\n Predictions")
    print(predictions[:,1])
    print(predictions[:,1].max(),predictions[:,1].min())
    print("\n Exp Predictions")
    print(np.exp(predictions[:,1]))
    print(np.exp(predictions[:,1]).max(),np.exp(predictions[:,1]).min())

    return params, x_pred, uncert_x_pred, params_pred, uncert_params_pred, diff, calibration

def show_nice_error_statistics(error_INN,uncert_error,name="CO2",position=position):

    plt.figure(f"Nice_errors_{name}", figsize=(15,5))
    #error_INN=np.clip(error_INN,-4,4)

    plot_boarders = max (np.max(error_INN), -np.min(error_INN))#4
    bins=np.linspace(-plot_boarders,plot_boarders,100)
    #error_ret=np.clip(error_ret ,-3,3)
    
    #Error distribution
    ax = plt.subplot(1, 3, 1)
    (mu_INN, sigma_INN)=norm.fit(error_INN)

    error_INN_clip=np.clip(error_INN,-plot_boarders,plot_boarders)
    ax.set_title(rf"Error distribution, $\mu$={mu_INN:.2f}, $\sigma$={sigma_INN:.2f}",fontsize=18)

    #plt.hist(error_ret, bins=bins, density=False, histtype='step',color="blue",label="retrival")
    plt.hist(error_INN_clip, bins=bins, density=False, histtype='step',color="blue",label="cINN")

    plt.xlabel("Estimated difference to gt in ppm",fontsize=18)
    plt.ylabel("Number of estimations",fontsize=20)

    ax = plt.subplot(1, 3, 2)

    uncert_bins=np.linspace(0,uncert_error.max(),100)
    ax.set_title(f"ACOS Mean:{ana_all[:,uncert_index].mean():.2f}, NN Mean {uncert_error.mean():.2f}",fontsize=15)
    plt.hist(uncert_error,bins = uncert_bins, density=False, histtype='step',color="blue",label="Neural Network")
    plt.hist(ana_all[:,uncert_index], bins=uncert_bins, density=False, histtype='step',color="brown",label="ACOS")
    plt.legend()

    plt.xlabel(r"1 $\sigma$ error estimate in ppm",fontsize=18)
    plt.ylabel("Number of estimations",fontsize=20)

    ax = plt.subplot(1, 3, 3)
    bins=np.linspace(-5,5,101)
    #bins=np.linspace(-10,10,100)
    rel_error_INN = (error_INN/uncert_error)
    #rel_error_INN=np.clip(rel_error_INN,-4,4)

    #plt.hist(error_ret/dataloader.ret_params[:c.evaluation_samples,1], density=True, bins=bins, histtype='step',color="blue",label="retrival")
    INN_bins,_,_ = plt.hist(rel_error_INN, bins=bins, histtype='step',density=True,color="green",label="cINN")

    (mu_INN, sigma_INN)=norm.fit(rel_error_INN)
    y_best = scipy.stats.norm.pdf( bins, 0, 1)
    xi2_INN = 0
    print(xi2_INN)
    y_INN = scipy.stats.norm.pdf( bins, mu_INN, sigma_INN)


    l = plt.plot(bins, y_best, color="black", linestyle="--", linewidth=2,label="optimal gaussian")
    l = plt.plot(bins, y_INN, 'r--', linewidth=2,label="gaussian fit on INN")
    #l = plt.plot(bins, y_ret, 'r--', linewidth=2)
    ax.set_title(fr"$\mu=${mu_INN:.2f}, $\sigma=$ {sigma_INN:.2f}",fontsize=18)#mu_ret: {mu_ret:.2f}, sigma_ret: {sigma_ret:.2f}, 
    c.mu.append(sigma_INN)
    c.sigma.append(mu_INN)
    plt.legend()
    plt.xlabel("Calibration: Error/Uncertainty",fontsize=18)
    plt.ylabel("Prob. density",fontsize=20)


    plt.tight_layout()


def main():
    np.random.seed(1000)
    torch.manual_seed(1000)

    model.eval()  
    params, x_pred, uncert_x_pred, params_pred, uncert_params_pred, diff, calibration = compute_uncert(x_all,y_all)
    plot_helpers.plot_good_map(np.abs(diff[:,c.co2_pos]),position, name=f"Total error CO2")
    plot_helpers.plot_good_map(np.abs(calibration[:,c.co2_pos]),position, name=f"Calibration: Error Uncertainty", title = f"Calibration: Error/Uncertainty")
    plot_helpers.plot_good_map(uncert_params_pred[:,c.co2_pos],position, name=r"1 $\sigma$ certainty of prediction")
    plot_helpers.plot_good_map(ana_all[:,uncert_index],position, name="Uncertainty physical retrieval")
    show_nice_error_statistics(diff[:,c.co2_pos],uncert_params_pred[:,c.co2_pos])
    
    if c.device == "cuda":
        torch.cuda.empty_cache()
        print("Empties cuda chache")
    plt.show(block=False)
    plot_helpers.save_results()

    input("Press enter to end!")
main()