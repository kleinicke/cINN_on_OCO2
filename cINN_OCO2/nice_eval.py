"""Evaluation plots

"""

import numpy as np

import matplotlib.pyplot as plt
import torch
from pathlib import Path

import config as c

import help_train as ht

import data.prepare_data as prepare_data
import data.dataloader as dataloader

import plot_helpers

import scipy
from scipy.stats import norm
import data.data_helpers as data_helpers

model = c.model
model.to(c.device)

model.load(c.filename)
model.eval()
print('Trainable parameters:')
print(sum([p.numel() for p in model.trainable_parameters]))

#Variables:
# how many different posteriors to show:
n_plots = 20

# how many dimensions of x to use:
n_x = c.x_dim#
N=c.x_dim
print(f"Evaluates with n_plots={n_plots}, n_x={n_x} and N={N}")


x_all, y_all, ana_all = data_helpers.concatenate_test_set(c.evaluation_samples)

position = prepare_data.y_to_spectra(y_all)[:,-2:]
uncert_index = prepare_data.ana_names.index("xco2_uncertainty")


########################################
#              Histogramms             #
########################################
N_post= c.N_post
z = torch.randn(N_post, c.x_dim).to(c.device)

def sample_posterior(y_it, z=z, N=N_post):
    """[summary]

    Arguments:
        y_it {[type]} -- [description]

    Keyword Arguments:
        z {[type]} -- [description] (default: {z})
        N {[type]} -- [description] (default: {N_post})

    Returns:
        [outputs] -- [sampled]
    """

    outputs = []
    print("samples_posterior",y_it.shape, z.shape)

    for i,y in enumerate(y_it):
            y = y.expand(N,-1)

            with torch.no_grad():
                outputs.append(model.reverse_sample(z, y).data.cpu().numpy())

    return outputs




def compute_calibration(x,y, load=False,save_name="test"):
    params=prepare_data.x_to_params(x)
    import time

    n_steps = 100
    q_values = []
    confidences = np.linspace(0., 1., n_steps+1, endpoint=True)[1:]
    for conf in confidences:

        q_low  = 0.5 * (1 - conf)
        q_high = 0.5 * (1 + conf)
        q_values += [q_low, q_high]
    print("Starts to sample posteriors for calibration errors")

    t = time.perf_counter()
    posterior = sample_posterior(y)
    diff = torch.mean(torch.FloatTensor(posterior), dim = 1)-x
    
    print("meandiff direct",torch.mean(diff, dim = 0))
    print("absmeandiff direct",torch.mean(torch.abs(diff), dim = 0))

    posterior=prepare_data.x_to_params(posterior)
    diff = np.mean(posterior, axis = 1) -params
    print("mean denorm",np.mean(diff, axis = 0))
    print("absmean denorm",np.mean(np.abs(diff), axis = 0))    
    diff3 = np.median(posterior, axis = 1) -params
    print("meandiff denorm median",np.mean(diff3, axis = 0))
    print("absmeandiff denorm median",np.mean(np.abs(diff3), axis = 0))     
    print(f"sample and x to params time: {time.perf_counter() - t:.2f}s")

    quantile=(np.quantile(posterior[:,:,:], q_values,axis=1))

    uncert_intervals=(quantile[1::2,:,:] - quantile[0::2,:,:])

    print(f"Total time to compute inlier matrix: {time.perf_counter() - t:.2f}s")


       


    inliers=((np.logical_and(np.less(params,quantile[1::2,:,:]),np.greater( params,quantile[0::2,:,:]))))
    uncert_error_co2=uncert_intervals[68,:,c.co2_pos]/2
    uncert_error=uncert_intervals[68,:,:]/2
    #Calculate error of INN
    #error_INN = (quantile[0,:,:]-gt_params)
    error_INN = (posterior.mean(axis=1)-params)
    return inliers,quantile,uncert_intervals,confidences, params, posterior, uncert_error_co2, uncert_error, error_INN

def hists(x, bins=40):
    results = []
    for j in range(N):
        h, b = np.histogram(x[:, j], bins=bins, density=True)#range=(-2,2),
        h /= np.max(h)
        results.append([b[:-1],h])
    return results

def show_posterior_histograms(params, posteriors, n_plots = n_plots, orig_prior_hists = hists(prepare_data.x_to_params(x_all))):


    confidence = 0.68
    q_low  = 100. * 0.5 * (1 - confidence)
    q_high = 100. * 0.5 * (1 + confidence)

    for i in range(n_plots):
        
        hist_i = hists(posteriors[i])
        
        show_orig_number=n_plots-10
        if i < show_orig_number:       
            plt.figure(f"orig_{i}",figsize=(15,7))
            for j in range(n_x):
                ax = plt.subplot((n_x+2)/3, 3, j +1)
                plt.step(*(orig_prior_hists[j]), where='post', color='grey',label="gt dist. of other samples") 
                plt.step(*(hist_i[j]), where='post', color='blue',label="sampled predictions")

                x_low, x_high = np.percentile(posteriors[i][:,j], [q_low, q_high])

                if j == c.co2_pos:
                    raw_index = prepare_data.ana_names.index("xco2_raw")
                    unc_index = prepare_data.ana_names.index("xco2_uncertainty")
                    ap_index = prepare_data.ana_names.index("xco2_apriori")

                    if x_low < 100:
                        raw_xco2 = ana_all[i,raw_index] - ana_all[i,ap_index]
                    else:
                        raw_xco2 = ana_all[i,raw_index]
                    #x_low=params[i,j]-ana_all[i,unc_index]
                    #x_high=params[i,j]+ana_all[i,unc_index]
                    ret_uncert = ana_all[i,unc_index]
                    ret_bins = np.linspace(params[i,j]-3*ret_uncert,params[i,j]+3*ret_uncert,100)
                    y_best = scipy.stats.norm.pdf( ret_bins, params[i,j], ret_uncert)

                    #scipy.stats.norm.pdf()
                    plt.plot(ret_bins, y_best/y_best.max(), color='brown',label="uncertainty of gt")

                plt.plot([x_low, x_low], [0,1], color='green')
                plt.plot([x_high, x_high], [0,1], color='green',label="1 sigma range")
                plt.plot([params[i,j], params[i,j]], [0,1], color='red',linewidth=2,label="ground truth")
                plt.plot([], [], color='brown',label="uncertainty of gt")

                unit = ""
                if c.param_names[j] == "xco2":
                    unit = " in ppm"
                if c.param_names[j] == "tcwv":
                    unit = r" in kg $m^{-2}$"
                #plt.xlabel(rf"{c.param_names[j]}{unit}")
                ax.set_title(rf"{c.param_names[j]}{unit}",fontsize=17)
            plt.legend(fontsize=13)
            plt.tight_layout()

        

def show_nice_calibration(inliers,uncert_intervals,confidences):
    inliers = np.mean(inliers, axis=1)
    uncert_intervals = np.median(uncert_intervals, axis=1)
    calib_err = inliers - confidences[:,None]
    uncert_intervals=uncert_intervals[:,c.co2_pos]
    calib_err=calib_err[:,c.co2_pos]
    plt.figure(f"nice_calibration_error_CO2",figsize=(10,5))
    plt.subplot(2, 1, 1)

    plt.plot(confidences, calib_err,label=c.param_names[c.co2_pos])
    plt.ylabel('Calibration error',fontsize=15)
    #plt.xlabel('Confidence')

    plt.xlim([0,1])
    plt.subplot(2, 1, 2)
    plt.plot(confidences, uncert_intervals/2)
    plt.ylabel('Median estimated uncertainty',fontsize=12)
    plt.xlabel('Confidence',fontsize=20)
    plt.xlim([0,1])
    #plt.legend()

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
    ax.set_title(f"ACOS Mean:{ana_all[:,uncert_index].mean():.2f}, NN Mean {uncert_error.mean():.2f}",fontsize=15)

    uncert_bins=np.linspace(0,uncert_error.max(),100)

    plt.hist(uncert_error,bins = uncert_bins, density=False, histtype='step',color="blue",label="cINN")
    plt.hist(ana_all[:,uncert_index], bins=uncert_bins, density=False, histtype='step',color="brown",label="ACOS")
    plt.legend()

    plt.xlabel(r"1 $\sigma$ error estimate in ppm",fontsize=18)
    plt.ylabel("Number of estimations",fontsize=20)

    ax = plt.subplot(1, 3, 3)
    bins=np.linspace(-4.5,4.5,100)
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
    plt.xlabel("Difference to gt, depending on estimated error",fontsize=18)
    plt.ylabel("Prob. density",fontsize=20)


    plt.tight_layout()


def show_nice_posterior_correlations(post, x_gt,prior_hists ,spectrum_number=0,x_start=0,x_end= c.x_dim, y_start = 0, y_end = c.x_dim,name="",show_distribution=False):
    plt.figure(f'{name}_correlations_{spectrum_number}',figsize=(15,12))
    #plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.title('correlations')
    hist_i = hists(post[spectrum_number])
    #for i in range(n_parameters-1,-1,-1):
    #    plt.subplot(n_parameters+1, n_parameters, i+1)
    #    plt.hist(orig_parameters[:, i+offset], bins=50, histtype='step')
    if show_distribution:
        for j in range(x_start,x_end):
            #ax=plt.subplot(n_parameters+1, n_parameters, j +1)
            ax=plt.subplot(y_end-y_start+1, x_end-x_start, j-x_start+1)

            plt.step(*(prior_hists[j]), where='post', color='grey')
            plt.step(*(hist_i[j]), where='post', color='blue')

            #x_low, x_high = np.percentile(post[i][:,j+offset], [q_low, q_high])
            plt.plot([x_gt[spectrum_number,j], x_gt[spectrum_number,j]], [0,1], color='red')
            #plt.plot([x_low, x_low], [0,1], color='orange')
            #plt.plot([x_high, x_high], [0,1], color='orange')
            plt.xlabel(f"{c.param_names[j]}")
            ax.xaxis.tick_top()
            #if j>0:
                #ax.axis('off')
            #    ax.set_yticks([])
            #if j==c.co2_pos:
            #    ax.set_xlim(375,390)
            if j>x_start:
                ax.set_yticks([])
            else:
                plt.ylabel(f"Prior dist", fontsize = 20)

    #for i in range(n_parameters-1,-1,-1):
    #for i in range(n_parameters):
    #    for j in range(n_parameters):
    offset = -y_start
    if show_distribution:
        offset = 1-y_start
    for j in range(y_start,y_end):
        for i in range(x_start,x_end):
            #print(i,j)
            #ax=plt.subplot(n_parameters+1, n_parameters, i+1+n_parameters*(j+1))
            ax=plt.subplot(y_end+offset, x_end-x_start, i-x_start+1+(x_end-x_start)*(j+offset))

            plt.subplots_adjust(hspace = .001, wspace=.001)
            plt.scatter(x=post[spectrum_number,:,i],y=post[spectrum_number,:,j],c="blue",s=0.5, alpha=0.1)
            plt.scatter(x=x_gt[spectrum_number,i],y=x_gt[spectrum_number,j],c='red',marker="+",s=150)
            if j<y_end-1:
                #ax.axis('off')
                ax.set_xticks([])
            else:
                plt.xlabel(c.param_names[i])

            if i>x_start:
                ax.set_yticks([])
            else:
                if len(c.param_names[j])>5:
                    plt.ylabel(f"{c.param_names[j][:3]}..{c.param_names[j][-3:]}")
                else:
                    plt.ylabel(f"{c.param_names[j]}")


            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                 #+ ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(20)


def main():
    np.random.seed(1000)
    torch.manual_seed(1000)

    model.eval()  

    inliers,quantile,uncert_intervals,confidences, params, posterior, uncert_error_co2, uncert_error, error_INN = compute_calibration(x_all,y_all)
    prior_hists = hists(params)

    show_posterior_histograms(params[:n_plots], posterior[:n_plots],orig_prior_hists = prior_hists)
    show_nice_calibration(inliers,uncert_intervals,confidences)
    print(error_INN.shape)

    plot_helpers.plot_good_map(np.abs(error_INN[:,c.co2_pos]),position, name=f"Total error CO2")
    plot_helpers.plot_good_map(np.abs(error_INN[:,c.co2_pos])/uncert_error[:,c.co2_pos],position, name=f"Relationship between actual and predicted error for CO2")
    plot_helpers.plot_good_map(uncert_error_co2,position, name=r"1 $\sigma$ certainty of prediction")
    plot_helpers.plot_good_map(ana_all[:,uncert_index],position, name="Uncertainty physical retrieval")
    show_nice_error_statistics(error_INN[:,c.co2_pos],uncert_error[:,c.co2_pos])
    for i in range(1):
        show_nice_posterior_correlations(posterior, params,prior_hists,spectrum_number=i)
        
    plot_helpers.show_prediction_changes(model)

    if c.device == "cuda":
        torch.cuda.empty_cache()
        print("Empties cuda chache")
    plt.show(block=False)
    plot_helpers.save_results()

    input("Press enter to end!")



main()