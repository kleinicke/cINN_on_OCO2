import numpy as np
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch

#import model
import config as c
#c.feature_net_file+="_trained"
c.additionally_trained_feature_net = False
c.use_pretrained = True 
#import new_net  as feature_net
import feed_forward_net as feature_net
#import easy_train as feature_net
import plot_helpers

import data.prepare_data as prepare_data
import data.dataloader as dataloader
import data.data_helpers as data_helpers

import scipy
from scipy.stats import norm
import matplotlib.mlab as mlab
#from scipy.stats import norm

model = feature_net.model
feature_net.load(c.feature_net_file)
model.eval()

#print('Trainable parameters:')
#print(sum([p.numel() for p in model.params_trainable]))

#Variables:
# how many different posteriors to show:
n_plots = 5

# how many dimensions of x to use:
n_x = c.x_dim#
offset=0
N=c.x_dim
print(f"Evaluates with n_plots={n_plots}, n_x={n_x}, offset={offset} and N={N}")

def concatenate_all_set(cut=None):
    x_all, y_all = [], []

    for x,y in c.test_loader:
        x_all.append(x)
        y_all.append(y)
    for x,y in c.train_loader:
        x_all.append(x)
        y_all.append(y)
    x_cat, y_cat = torch.cat(x_all, 0), torch.cat(y_all, 0)

    if cut:
        x_cat, y_cat = x_cat[:cut], y_cat[:cut]
    return x_cat, y_cat

def concatenate_train_set(cut=None):
    x_all, y_all = [], []

    for x,y in c.train_loader:
        x_all.append(x)
        y_all.append(y)

    x_cat, y_cat = torch.cat(x_all, 0), torch.cat(y_all, 0)

    if cut:
        x_cat, y_cat = x_cat[:cut], y_cat[:cut]
    return x_cat, y_cat

def concatenate_test_set(cut=None):
    x_all, y_all, ana_all = [], [],[]

    for idx, (x,y, ana) in enumerate(c.test_ana_loader): #,ana
        x_all.append(x)
        y_all.append(y)
        #print("\nx\n",model.fn_func(y.to(c.device)).detach().cpu().numpy())
        ana_all.append(ana)
        if cut:
            if idx*c.batch_size>cut:
                break

    x_cat, y_cat, ana_cat = torch.cat(x_all, 0), torch.cat(y_all, 0), torch.cat(ana_all, 0) #ana_all

    if cut:
        x_cat, y_cat, ana_cat = x_cat[:cut], y_cat[:cut], ana_cat[:cut]
    return x_cat, y_cat, ana_cat

x_all, y_all, ana_all = concatenate_test_set(c.evaluation_samples)#concatenate_test_set(1000)#400)#20)#400)

ana_all = ana_all.detach().cpu().numpy()

position = prepare_data.y_to_spectra(y_all)[:,-2:]
#position=dataloader.ret_params[:c.evaluation_samples,[-2,-3]] 


def hists(x):

    results = []
    for j in range(N):
        h, b = np.histogram(x[:, j], bins=100, density=True)#range=(-2,2),
        h /= np.max(h)
        results.append([b[:-1],h])
    return results


def show_error_stats(errors, name = "CO2", show_nice = False):
    plt.figure(name,figsize=(8,2))

    #plt.title(f"Total {name} error")
    border = max(np.abs(errors.min()),np.abs(errors.max()))
    bins=np.linspace(-border,border,100)

    #plt.hist(error_ret, bins=bins, density=False, histtype='step',color="blue",label="retrival")
    plt.hist(errors, bins=bins, density=True, histtype='step',color="blue")

    (mu, sigma)=norm.fit(errors)
    y_fn = scipy.stats.norm.pdf(bins, mu, sigma)

    plt.plot(bins, y_fn, 'r--', linewidth=2)
    plt.title(rf"$\mu$ = {mu:.2f}, $\sigma$ = {sigma:.2f}")
    plt.xlabel("Difference to gt in ppm")
    plt.ylabel("Prob. density")
    #plt.legend()
    plt.tight_layout()
    
    if show_nice:
        plt.figure("nice_nn_sol",figsize=(16,5))
        plt.subplot(1, 3, 1)
        plt.hist(errors, bins=bins, density=True, histtype='step',color="blue")

        (mu, sigma)=norm.fit(errors)
        y_fn = scipy.stats.norm.pdf(bins, mu, sigma)

        plt.plot(bins, y_fn, 'r--', linewidth=2)
        plt.title(rf"$\mu$ = {mu:.2f}, $\sigma$ = {sigma:.2f}",fontsize = 18)
        plt.xlabel("Difference to gt in ppm",fontsize = 18)
        plt.ylabel("Prob. density",fontsize = 18)
        plt.tight_layout()
        
    
def show_predicted_error():
    pass

def show_feature_net_solution():
    print("show_feature_net_solution")
    orig_prior_hists = hists(prepare_data.x_to_params(x_all))

    x = model.fn_func(y_all.to(c.device)).detach().cpu().numpy()
    #print("co2_results",x[:100,0])
    #print("true_results",x_all[:100,0])
    
    #y_test = y_all.numpy()+prepare_data.mu_y
    #print(y_all[:10,:10], y_all[:10,-10:])
    y_gt = y_all#[:n_plots]
    x_gt = x_all
    orig_x_gt = prepare_data.x_to_params(x_gt)
    orig_y_gt = prepare_data.y_to_spectra(y_gt)
    #print(x.shape)
    orig_x=prepare_data.x_to_params(x)
    #print(np.shape(orig_x), np.shape(orig_x_gt))
    #print("\n")
    #plot_world.plot_quality_map((np.abs(orig_x-orig_x_gt)[:,0]),(y_all.detach().cpu().numpy()+prepare_data.mu_y)[:,-2:], "Featurenet prediction")
    plot_helpers.plot_quality_map((np.abs(orig_x-orig_x_gt)[:,0]),position, "Error of network prediction")

    #print(x)
    #print(x_gt)
    print("shapes:",orig_x.shape,orig_x_gt.shape)
    show_error_stats(orig_x[:,0]-orig_x_gt[:,0], show_nice= True)

    print("show_predicted_error")

    x_uncert = model.fn_func(y_all.to(c.device)).detach().cpu().numpy()[:,1]
    #compute s to sigma, following the paper
    x_uncert = np.sqrt(np.exp(x_uncert))

    uncert = x_uncert*np.linalg.inv(prepare_data.w_x)[0,0]

    plot_helpers.plot_quality_map(uncert,position, "Predicted Uncertainty")

    #show_error_stats(uncert, "uncertainty")

    #show_error_stats((orig_x[:,0]-orig_x_gt[:,0])/uncert, "normalized")

    plot_helpers.plot_quality_map(np.abs((orig_x[:,0]-orig_x_gt[:,0])/uncert),position, "Uncertainty quality")



    #orig_x_gt = orig_x_gt[:n_plots]
    #x_gt = x_gt[:n_plots]
    for i in range(n_plots):
        #print(x_gt[0])
        #print("\n",prepare_datax_to_params(x_gt)[0])
        #print(prepare_datax_to_params(x)[0])
        #print(x[0])

        plt.figure(f"orig_{i}",figsize=(20,15))
        for j in range(n_x):
            plt.subplot(3, n_x/4+1, j +1)
            if j == 0:
                plt.step(*(orig_prior_hists[j]), where='post', color='grey',label= "prior") 
                plt.plot([orig_x_gt[i,j], orig_x_gt[i,j]], [0,1], color='red', label = "ground truth")
                plt.plot([orig_x[i,j], orig_x[i,j]], [0,1], color='blue', label = "predicted value")
                plt.legend()
            else:
                plt.step(*(orig_prior_hists[j]), where='post', color='grey') 
                #plt.step(*(hist_i[j+offset]), where='post', color='blue')

                #x_low, x_high = np.percentile(orig_posteriors[i][:,j+offset], [q_low, q_high])
                plt.plot([orig_x_gt[i,j], orig_x_gt[i,j]], [0,1], color='red')
                plt.plot([orig_x[i,j], orig_x[i,j]], [0,1], color='blue')
            #plt.plot([orig_y_gt [i,j+offset-18], orig_y_gt[i,j+offset-18]], [0,1], color='orange',alpha=0.5) #is on top of red
            #if j+offset == 14:
            #    x_low=dataloader.ret_params[i,0]-dataloader.ret_params[i,1]
            #    x_high=dataloader.ret_params[i,0]+dataloader.ret_params[i,1]
            #    plt.plot([x_low, x_low], [0,1], color='green')
            #    plt.plot([x_high, x_high], [0,1], color='green')
            plt.xlabel(f"{c.param_names[j]}")
        #plt.legend()
        #plt.tight_layout()


def show_prediction_changes():
    #torch.manual_seed(71)
    years = c.dc.viz_years
    param_gt_mean = []
    output_param_mean = []
    #batch_size = 512
    #for _, year in enumerate(years):
    for i,loadset in enumerate(dataloader.loader):
        year = c.dc.viz_years[i]
        #sets = dataloader.loadOCODataset(year = [year], analyze=True, noise=False)
        #loadset = dataloader.DataLoader(sets,
        #    batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 1)
        x,y,_ = data_helpers.concatenate_set(loadset,1000)
        output = model.fn_func(y.to(c.device)).detach().cpu().numpy()
        #output = torch.mean(torch.FloatTensor(outputs), dim = 1)
        #print(output.size())
        #with torch.no_grad():
        #    output = model.fn_func(y.to(c.device)).detach().cpu()
        
        #errors = ht.show_error(f"{year} eval", visualize=False)
        #errors.add_error(output,x)
        #errors.print_error()
        param_gt = prepare_data.x_to_params(x)
        output_param = prepare_data.x_to_params(output)
        param_gt_mean.append(np.mean(param_gt[:,0]))
        output_param_mean.append(np.mean(output_param[:,0]))
    plt.figure("Cmp_pred_true_INN")
    plt.plot(years, param_gt_mean, label = "gt")
    plt.plot(years, output_param_mean, label = "prediction_INN")
    plt.legend()
    plt.title("Comparison between predicted and true CO2 concentration")
    plt.xlabel("year")
    plt.ylabel("CO2 in ppm")

    plt.figure("Offset_per_year_INN")
    plt.title("Offset per year")
    plt.plot(years, np.subtract(output_param_mean,param_gt_mean), label = "offset")
    plt.xlabel("year")
    plt.ylabel("CO2 in ppm")
    plt.legend()

    plt.figure("Increases_per_year_fn")
    plt.title("Increases per year")
    plt.plot(years[1:], np.diff(output_param_mean), label = "increase prediction")
    plt.plot(years[1:], [j-i for i, j in zip(param_gt_mean[:-1], param_gt_mean[1:])], label = "increase gt")
    plt.xlabel("year")
    plt.ylabel("CO2 in ppm")
    plt.legend()

    plt.figure("nice_nn_sol",figsize=(14,5))
    
    plt.subplot(1, 3, 2)
    plt.plot(years, param_gt_mean, label = "True value")
    plt.plot(years, output_param_mean, label = "Prediction ")
    plt.legend(fontsize = 18)
    #plt.title("Comparison between predicted and true CO2 concentration")
    plt.title("Feedforward network",fontsize = 18)
    plt.xlabel("year",fontsize = 18)
    plt.ylabel("CO2 in ppm",fontsize = 18)
    
    plt.subplot(1, 3, 3)
    plt.title("Offset per year",fontsize = 18)
    plt.plot(years, np.subtract(output_param_mean,param_gt_mean), label = "offset")
    plt.xlabel("year",fontsize = 18)
    plt.ylabel("CO2 in ppm",fontsize = 18)
    #plt.legend(fontsize = 18)
    plt.tight_layout()

def uncert(model2=model,name="CO2",position=position):
    x2 = model2.fn_func(y_all.to(c.device)).detach().cpu().numpy()
    params2 = prepare_data.x_to_params(x2[:,::2])[:,0]
    params_gt = prepare_data.x_to_params(x_all)[:,0]
    errors2 = params2 - params_gt
    uncert2 = prepare_data.x_to_params(np.sqrt(np.exp(x2[:,1::2])),no_mu=True)[:,0]
    calib2 = errors2/uncert2


    plt.figure(f"Nice_uncerts_{name}", figsize=(15,5))
    #error_INN=np.clip(error_INN,-4,4)

    plot_boarders = max (np.max(errors2), -np.min(errors2))#4
    bins=np.linspace(-plot_boarders,plot_boarders,100)
    #error_ret=np.clip(error_ret ,-3,3)
    
    #Error distribution
    ax = plt.subplot(1, 2, 1)
    (mu_INN, sigma_INN)=norm.fit(errors2)

    errors2_clip=np.clip(errors2,-plot_boarders,plot_boarders)
    uncert2_clip=np.clip(uncert2,-plot_boarders,plot_boarders)
    ax.set_title(rf"Error distribution, $\mu$={mu_INN:.2f}, $\sigma$={sigma_INN:.2f}",fontsize=20)

    #plt.hist(error_ret, bins=bins, density=False, histtype='step',color="blue",label="retrival")
    plt.hist(errors2_clip, bins=bins, density=False, histtype='step',color="blue",label="cINN")
    plt.hist(uncert2_clip, bins=bins, density=False, histtype='step',label="uncert")

    plt.xlabel("Estimated difference to gt in ppm",fontsize=20)
    plt.ylabel("Number of estimations",fontsize=20)

    

    ax = plt.subplot(1, 2, 2)
    bins=np.linspace(-4.5,4.5,100)
    #rel_error_iNN=np.clip(rel_error_iNN,-4,4)

    #plt.hist(error_ret/dataloader.ret_params[:c.evaluation_samples,1], density=True, bins=bins, histtype='step',color="blue",label="retrival")
    iNN_bins,_,_ = plt.hist(calib2, bins=bins, histtype='step',density=True,color="blue",label="cINN")




    (mu_INN, sigma_iNN)=norm.fit(calib2)
    y_best = scipy.stats.norm.pdf( bins, 0, 1)
    xi2_iNN = 0#np.sum(np.square(y_best-iNN_bins)/y_best)
    print(xi2_iNN)
    y_INN = scipy.stats.norm.pdf( bins, mu_INN, sigma_iNN)


    l = plt.plot(bins, y_best, color="black", linestyle="--", linewidth=2,label="optimal gaussian")
    l = plt.plot(bins, y_INN, 'b--', linewidth=2,label="gaussian fit on INN")
    #l = plt.plot(bins, y_ret, 'r--', linewidth=2)
    ax.set_title(f"mu_INN:{mu_INN:.2f}, sigma_iNN: {sigma_iNN:.2f}",fontsize=20)#mu_ret: {mu_ret:.2f}, sigma_ret: {sigma_ret:.2f}, 
    c.mu.append(sigma_iNN)
    c.sigma.append(mu_INN)
    plt.legend()
    plt.xlabel("Difference to gt, depending on estimated error",fontsize=20)
    plt.ylabel("Prob. density",fontsize=20)
    plt.tight_layout()

#TODO: Show whats the error for what percentage of samples.
def main():
    show_predicted_error()
    show_feature_net_solution()
    show_prediction_changes()
    #uncert()
    if c.device == "cuda":
        torch.cuda.empty_cache()
        print("Empties cuda chache")

    plot_helpers.save_results()
    plt.show(block=False)
    input("Press enter to end")

main()