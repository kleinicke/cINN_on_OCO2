import geopandas
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import data.prepare_data as prepare_data
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors


def plot_quality_map(value,position,name="Satelite positon",title="", mask = None):
    """AKA world map

    
    Arguments:
        uncert_intervals {[type]} -- [description]
    
    Keyword Arguments:
        name {str} -- [description] (default: {"Satelite positon"})
        title {str} -- [description] (default: {"Width of uncertanty, depending on position"})
    """

    #external_results=prepare_prepare_dataco2_ret_params[:,-4:]
    #external_results=dataloader.ret_params[:c.evaluation_samples,-4:]
    #print(error.shape)
    print("Mean error is for worldmap is:",np.mean(value))
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
    fig = plt.figure(name,figsize=(20,15))
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
    world.plot(ax=ax, color='lightblue', edgecolor='black')
    pos = position
    #print(pos)
    levels = np.percentile(value[mask], np.linspace(0,100,101))
    norm = matplotlib.colors.BoundaryNorm(levels,256)
    im = plt.scatter(pos[:,0][mask],pos[:,1][mask],c=value[mask],s=15, norm=norm)#,cmap=plt.get_cmap("jet"))#"plasma"
        #plt.colorbar(im)
    #plt.legend(title="in ppm")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")   
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.08)
    ticks = np.percentile(value[mask], np.linspace(0,100,9))
    plt.colorbar(im, cax=cax,aspect=20,ticks=ticks)
    #plt.show()
    #ax.colorbar(im, aspect=20)
    #plt.colorbar()


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
