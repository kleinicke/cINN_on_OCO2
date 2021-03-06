import geopandas
import matplotlib
import torch
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import data.prepare_data as prepare_data
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors
from pathlib import Path
import config as c
import data.dataloader as dataloader
import data.data_helpers as data_helpers



def plot_quality_map(value,position,name="Satelite positon",title="", mask = None):
    """AKA world map

    
    Arguments:
        uncert_intervals {[type]} -- [description]
    
    Keyword Arguments:
        name {str} -- [description] (default: {"Satelite positon"})
        title {str} -- [description] (default: {"Width of uncertainty, depending on position"})
    """
    #value *= 1.25
    #external_results=prepare_prepare_dataco2_ret_params[:,-4:]
    #external_results=dataloader.ret_params[:c.evaluation_samples,-4:]
    #print(error.shape)
    print("Mean error is for worldmap is:",np.mean(value))
    print(value,position)
    print(value.shape,position.shape)
    #error=np.clip(error,0,2.5)
    if title == "":
        title=name
    #co2_mask=np.logical_and(np.greater_equal(error,377),np.less(error,389))
    #month_mask1 = np.isclose(external_results[:,0],1,1e-3)#[co2_mask]
    #month_mask4 = np.isclose(external_results[:,0],4,1e-3)#[co2_mask]
    #month_mask7 = np.isclose(external_results[:,0],7,1e-3)#[co2_mask]
    #month_mask10 = np.isclose(external_results[:,0],10,1e-3)#[co2_mask]
    #month_masks = [month_mask1,month_mask4,month_mask7,month_mask10]
    #month = ["january","april","july","october"]
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    #if mask is None:
    #    mask = np.ones(error.shape,dtype=bool)
    fig = plt.figure(name,figsize=(20,15))#figsize=(13,9))#
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
def plot_good_map(value,position,name="Satelite positon",title=""):
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

    levels = np.percentile(value, np.linspace(0,100,101))
    norm = matplotlib.colors.BoundaryNorm(levels,256)
    print(position.shape, value.shape)
    im = plt.scatter(position[:,0],position[:,1],c=value,s=40, norm=norm,edgecolors='black')#,cmap=plt.get_cmap("jet"))#"plasma",cmap='hot'#s=15

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")   
    

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.08)
    ticks = np.percentile(value, np.linspace(0,100,9))
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

    plt.hist(uncert_error,bins = uncert_bins, density=False, histtype='step',color="blue",label="iNN")
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


