"""These functions are used for evaluation

"""

import geopandas
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import data.prepare_data as prepare_data
import data.data_helpers as data_helpers
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors
from pathlib import Path
import config as c
import torch
import data.dataloader as dataloader




def plot_quality_map(value,position,name="Satelite positon",title="", mask = None):
    """AKA world map

    
    Arguments:
        uncert_intervals {[type]} -- [description]
    
    Keyword Arguments:
        name {str} -- [description] (default: {"Satelite positon"})
        title {str} -- [description] (default: {"Width of uncertainty, depending on position"})
    """

    print("Mean error is for worldmap is:",np.mean(value))
    if title == "":
        title=name

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    fig = plt.figure(name,figsize=(20,10))
    #plt.tight_layout()
    #plt.title(title)
    ax = plt.subplot(1,1,1)
    #fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(20,15))
    #fig.tight_layout()
    #fig.set_title(title)
    #print(np.shape(external_results), np.shape((error)), np.shape(month_masks))
        #ax = plt.subplot(2, 2, 1)
    ax.set_aspect('equal')
    ax.set_title(f"{title}")
    world.plot(ax=ax, color='lavender', edgecolor='black')#lightblue,burlywood

    #print(pos)
    levels = np.percentile(value[mask], np.linspace(0,100,101))
    norm = matplotlib.colors.BoundaryNorm(levels,256)
    im = plt.scatter(position[:,0][mask],position[:,1][mask],c=value[mask],s=15, norm=norm)#,cmap=plt.get_cmap("jet"))#"plasma",cmap='hot'
        #plt.colorbar(im)
    #plt.legend(title="in ppm")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")   
    

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.08)
    ticks = np.percentile(value[mask], np.linspace(0,100,9))
    cbar  = plt.colorbar(im, cax=cax,aspect=20,ticks=ticks)
    cbar.set_label("Uncertainty in ppm")
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, cbar.ax.yaxis.label] +
    ax.get_xticklabels() + ax.get_yticklabels()+cbar.ax.get_yticklabels()):
        item.set_fontsize(23)
    plt.tight_layout()



    #plt.show()
    #ax.colorbar(im, aspect=20)
    #plt.colorbar()
    plot_nice_world(value,position,name=name,title=title)

    """
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    ax = plt.gca()
    im = ax.imshow(np.arange(100).reshape((10,10)))

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    """

def plot_good_map(value,position,name="Satelite positon",title="", mask = None):
    """AKA world map

    
    Arguments:
        uncert_intervals {[type]} -- [description]
    
    Keyword Arguments:
        name {str} -- [description] (default: {"Satelite positon"})
        title {str} -- [description] (default: {"Width of uncertainty, depending on position"})
    """
    
    print("Mean error is for worldmap is:",np.mean(value))
    if title == "":
        title=name

    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    fig = plt.figure(name,figsize=(20,12))#figsize=(13,9))#

    ax = plt.subplot(1,1,1)

    ax.set_aspect('equal')
    ax.set_title(f"{title}")
    world.plot(ax=ax, color='azure', edgecolor='black')#lightblue,burlywood,lavender,aliceblue

    levels = np.percentile(value[mask], np.linspace(0,100,101))
    norm = matplotlib.colors.BoundaryNorm(levels,256)
    im = plt.scatter(position[:,0][mask],position[:,1][mask],c=value[mask],s=40, norm=norm,edgecolors='black')#,cmap=plt.get_cmap("jet"))#"plasma",cmap='hot'#s=15

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")   
    

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.08)
    ticks = np.percentile(value[mask], np.linspace(0,100,9))
    cbar  = plt.colorbar(im, cax=cax,aspect=20,ticks=ticks)
    cbar.set_label("in ppm")
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, cbar.ax.yaxis.label] +
    ax.get_xticklabels() + ax.get_yticklabels()+cbar.ax.get_yticklabels()):
        item.set_fontsize(23)
    plt.tight_layout()



def plot_nice_world(value,position,name="Satelite positon",title="", mask = None):
    if title == "":
        title=name


    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    xedges = np.linspace(-180,180,2*90+1)#361
    yedges = np.linspace(-90,90,2*45+1)#181

    H_sum, _, _ = np.histogram2d(position[:,0],position[:,1], bins=(xedges,yedges))
    H_value, _, _ = np.histogram2d(position[:,0],position[:,1], bins=(xedges,yedges), weights=value)
    #print(H_value[40:70,10:30])
    #print(H_sum[40:70,10:30])

    H_value = H_value/(H_sum)
    #print(H_value[40:70,10:30])

    levels = np.nanpercentile(H_value.flatten(), np.linspace(0,100,101))
    #print("levels",levels)
    norm = matplotlib.colors.BoundaryNorm(levels,256)

    fig = plt.figure(name+"nice",figsize=(14,7))
    ax = plt.subplot(1,1,1)
    ax.set_aspect('equal')
    ax.set_title(f"{title}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")   
    im = plt.imshow(H_value.T,origin='lower',extent=(-180,180,-90,90),alpha=1, norm=norm,cmap='winter_r')
    world.plot(ax=ax, color='lavender', edgecolor='black')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.08)
    ticks = np.nanpercentile(H_value, np.linspace(0,100,9))
    #print("ticks",np.nanpercentile(H_value, np.linspace(0,100,9)))
    #print(H_value.T[40:70,10:30])
    #print("\n")
    c = plt.colorbar(im, cax=cax,aspect=20,ticks=ticks)
    c.set_label("Uncertainty in ppm")
    plt.tight_layout()


def plot_nice_error_statistics(uncert_error):
    plt.figure("Error_estimate_total")#"nice error stat")
    plt.title("Sigmas of INN distributions")
    uncert_bins=np.linspace(0,uncert_error.max(),100)

    plt.hist(uncert_error*1.25,bins = uncert_bins, density=False, histtype='step',color="blue",label="iNN")
    #ax.set_xlabel("68% confidence intervall in ppm")
    #ax.set_ylabel("Number of estimateions")
    plt.xlabel(r"1 $\sigma$ error estimate in ppm")
    plt.ylabel("Number of estimations")
    plt.tight_layout()

    #plt.legend()

def plot_nice_posterior_histograms(hists,orig_posteriors,n_plots,orig_prior_hists,orig_x_gt,params):
    j=0
    confidence = 0.68
    q_low  = 100. * 0.5 * (1 - confidence)
    q_high = 100. * 0.5 * (1 + confidence)
    orig_prior_hists = hists(params,bins = 20)
    for i in range(n_plots):
        hist_i = hists(orig_posteriors[i])

        plt.figure(f"nice_orig_{i}")
        plt.step(*(orig_prior_hists[j]), where='post', color='grey', label = "all 2018 measurements") 
        plt.step(*(hist_i[j]), where='post', color='blue', label = "Samples of one spectrum")

        x_low, x_high = np.percentile(orig_posteriors[i][:,j], [q_low, q_high])
        plt.plot([orig_x_gt[i,j], orig_x_gt[i,j]], [0,1], color='red', label = "ground truth")
        plt.plot([x_low, x_low], [0,1], color='green', label = r"1 $\sigma$ environment")
        plt.plot([x_high, x_high], [0,1], color='green')

        #plt.xlabel(rf"{c.param_names[j]}{unit}")
        #ax.set_title(rf"{c.param_names[0]}{unit}")
        plt.xlabel(rf"CO$_2$ in ppm")
        plt.ylabel(f"probability density")
        plt.title("One measurement")
        plt.legend()
        plt.tight_layout()

    
def show_prediction_changes(model, show_two = False):
    N_post= c.N_post
    z = torch.randn(N_post, c.x_dim).to(c.device)
    model.eval()
    def sample_posterior(y_it, z=z, N=N_post,model=model):
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

    import help_train as ht
    three = True
    #import nice_eval as ne
    torch.manual_seed(71)
    years = c.dc.viz_years#[2014, 2015, 2016, 2017, 2018]
    param_gt_mean = []
    param_gt_mean1 = []
    param_gt_mean2 = []
    param_gt_mean3 = []
    other_gt_mean = []
    output_param_mean = []
    other_param_mean = []
    #batch_size = 512
    #for _, year in enumerate(years):
    for i,year in enumerate(dataloader.dc.viz_years):
        x,y = dataloader.year_sets[year]
        x.to(c.device)
        y.to(c.device)
        print(f"Creates prediction for year {c.dc.viz_years[i]}\n\n\n\n")
        #year = c.dc.viz_years[i]
        #sets = dataloader.loadOCODataset(year = [year], analyze=True, noise=False)
        #loadset = dataloader.DataLoader(sets,
        #    batch_size=batch_size, shuffle=True, drop_last=True, num_workers = 1)
        #x,y,_ = data_helpers.concatenate_set(loadset,1000)'
        #print(i,year_set)
        #x,y = year_set
        print(x.shape, y.shape)
        outputs = sample_posterior(y)
        output = torch.mean(torch.FloatTensor(outputs), dim = 1)
        #print(output.size())
        #with torch.no_grad():
        #    output = feature_net.model.fn_func(y.to(c.device)).detach().cpu()
        
        #errors = ht.show_error(f"{year} eval", visualize=False)
        #errors.add_error(output,x)
        #errors.print_error()
        param_gt = prepare_data.x_to_params(x)
        output_param = prepare_data.x_to_params(output)
        param_gt_mean.append(np.mean(param_gt[:,0]))
        if three:
            param_gt_mean1.append(np.mean(param_gt[:,0]))
            param_gt_mean2.append(np.mean(param_gt[:,1]))
            param_gt_mean3.append(np.mean(param_gt[:,2]))
        
        output_param_mean.append(np.mean(output_param[:,0]))
        if show_two:
            other_gt_mean.append(np.mean(param_gt[:,1]))
            other_param_mean.append(np.mean(output_param[:,1]))
    
    param_gt_mean1 = np.array(param_gt_mean1)
    param_gt_mean2 = np.array(param_gt_mean2)
    param_gt_mean3 = np.array(param_gt_mean3)
    
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

    plt.figure("nice_yoy",figsize=(14,5))
    ax = plt.subplot(1,2,1)
    plt.plot(years, param_gt_mean, label = "True value apriori")
    plt.plot(years, output_param_mean, label = "Prediction apriori")
    if show_two:
        plt.plot(years, other_gt_mean, label = "True value CO2")
        #plt.plot(years, other_param_mean, label = "Prediction co2")
    plt.legend(fontsize = 18)
    plt.title("Comparison between predicted and true mean CO2 concentrations")
    plt.xlabel("year",fontsize = 18)
    plt.ylabel("mean CO2 concentration in ppm",fontsize = 18)
    ax = plt.subplot(1,2,2)
    plt.title("Differences from true mean",fontsize = 18)
    plt.plot(years, np.subtract(output_param_mean,param_gt_mean),label = "apriori")
    if show_two:
        plt.plot(years, np.subtract(other_param_mean,other_gt_mean),label = "Measured")
    plt.xlabel("year",fontsize = 18)
    plt.ylabel("difference of mean CO2 in ppm",fontsize = 18)
    if show_two:
        plt.legend()
    plt.tight_layout()
    if three:
        plt.figure("raw_aprioi_xco2",figsize=(14,5))
        ax = plt.subplot(1,2,1)
        plt.title("Mean Concentrations",fontsize = 18)
        plt.plot(years, param_gt_mean1, label = "A Priori CO2")
        plt.plot(years, param_gt_mean2, label = "Raw CO2")
        plt.plot(years, param_gt_mean3, label = "Correct CO2")
        plt.xlabel("year",fontsize = 18)
        plt.ylabel("CO2 in ppm",fontsize = 18)
        plt.legend(fontsize = 18)
        
        ax = plt.subplot(1,2,2)
        plt.title("Differences in Concentrations",fontsize = 18)
        plt.plot(years, param_gt_mean1 - param_gt_mean2, label = "A Priori - Raw")
        plt.plot(years, param_gt_mean3 - param_gt_mean2, label = "Correct - Raw")
        plt.plot(years, param_gt_mean3 - param_gt_mean1, label = "Correct - A Priori")
        plt.xlabel("year",fontsize = 18)
        plt.ylabel("CO2 in ppm",fontsize = 18)
        plt.legend(fontsize = 18)
\

def save_results():
    image_path = Path(__file__).parent.joinpath("images/")
    #print(image_path)
    #print(image_path.joinpath("hi/"))
    i = 1
    while image_path.joinpath(f"{i}").exists():
        i=i+1
    
    print(i)
    image_path = image_path.joinpath(f"{i}")
    print(f"will save at {image_path}.")
    image_path.mkdir()
    figs = [plt.figure(n) for n in plt.get_fignums()]
    labels = [n for n in plt.get_figlabels()]

    for i, fig in enumerate(figs):
        fig.savefig(image_path.joinpath(f"{labels[i]}.pdf"),format='pdf')

    config_str = ""
    config_str += "==="*30 + "\n"
    config_str += "Data Config options:\n\n"
    for v in dir(c):
        if v[0]=='_': continue
        s=eval('c.%s'%(v))
        config_str += "  {:25}\t{}\n".format(v,s)

    config_str += "\n"
    config_str += "==="*30 + "\n"
    config_str += "Config options:\n\n"

    for v in dir(c):
        if v[0]=='_': continue
        s=eval('c.%s'%(v))
        config_str += "  {:25}\t{}\n".format(v,s)

    config_str += "==="*30 + "\n"

    with image_path.joinpath(f"settings{c.configuration}_{c.run_name}.txt").open("w", encoding ="utf-8") as f:
        f.write(config_str)


