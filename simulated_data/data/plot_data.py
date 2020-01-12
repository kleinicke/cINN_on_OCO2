import numpy as np
import glob
import re
try:
    import data.prepare_data as prepare_data
    from data.prepare_data import wavelenth_y,output_y,output_x,element_names,output_params,normizing_factors
    from data.prepare_data import mu_x,mu_y,w_x,w_y, all_element_names, external_params
    print("loaded data.prepare_data")
except:
    import prepare_data 
    from prepare_data import wavelenth_y,output_y,output_x,element_names,output_params,normizing_factors
    from prepare_data import mu_x,mu_y,w_x,w_y, all_element_names, external_params
    print("loaded prepare_data")

#import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))

co2_pos = len(prepare_data.output_params[0])-4

"""Plotting of input data histogramms and qualitative checking of single spectra and parameter
"""
#import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 
n_parameters=prepare_data.parameter_size#29#30#9#parameter_size
n_observ=5
offset=0
output_x=prepare_data.output_x[:,:]
output_params=prepare_data.output_params[:,:]/normizing_factors
log_spectra = prepare_data.output_spects
params_in_spectrum = prepare_data.number_external_params #5 or 18

def show_param_hists(parameters,n_parameters=n_parameters,offset=0,name=""):
        plt.figure(f'{name} parameter histograms')
        plt.title('parameter histograms')
        for i in range(n_parameters):
            plt.subplot(5, 4, i+1)
            print(parameters[:, i+offset])
            plt.hist(parameters[:, i+offset], bins=50, histtype='step')

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

def show_one_input_data():
    """
    Plots certain parameter of the first input example.
    """

    plt.figure("whighened")
    plt.plot(wavelenth_y[0,:],output_y[0,:])
    plt.figure("original")
    plt.plot(wavelenth_y[0,:],output_params[0,:])
    plt.figure("y_log")
    plt.plot(wavelenth_y[0,:],np.log(output_params[0,:]))
    #print(np.log(y_long[0,:]),y_long[0,:])
    print(np.shape(mu_y),np.shape(w_y))
    print(mu_y[0],w_y[0])
    #plt.figure("retrived")
    #plt.plot(wavelenth_y[0,:],np.log(retreved[0,:]))
    #plt.figure("retrived2")
    #plt.plot(wavelenth_y[0,:],np.log(retreved2[0,:]))
    plt.figure("x_long[0,:]")
    plt.plot(range(np.shape(output_x[0,:])[0]),output_x[0,:])
    #plt.yscale("log")
    plt.figure("params[0,3:31]")
    plt.plot(range(np.shape(output_params[0,:])[0]),output_params[0,:])
    plt.yscale("log")
    print(output_params[0,:])

def show_spectra():
    for i in [0,5,10,15,18,20,24,28]:
        i=i*1000
        if len(log_spectra) > i:
            plt.figure(f"spectra {i}")
            plt.plot(np.exp(wavelenth_y[0,:-params_in_spectrum]),np.exp(log_spectra[i,:-params_in_spectrum]))
            print("\n")
            print("spectrum ",i)
            print(wavelenth_y[0,:-params_in_spectrum], np.shape(wavelenth_y[0,:-params_in_spectrum]))
            print(log_spectra[i,:-params_in_spectrum], np.shape(log_spectra[0,:-params_in_spectrum]))
            print(log_spectra[i,-params_in_spectrum:])
            print(output_params[i,:])

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
        corr[i]=output_x[:20,i+1]/output_x[:20,0]
    print(corr)

def show_nice_elementcorrelations(parameters=output_params,x_start=13,x_end= n_parameters, y_start = 14, y_end = n_parameters,name="",element_names=element_names,show_distribution = False):

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
    if show_distribution:
        for i in range(x_end-1,x_start-1,-1):
            ax=plt.subplot(y_end-y_start+1, x_end-x_start, i-x_start+1)
            bins=np.linspace(np.min(parameters[:, i]),np.max(parameters[:, i]),100)
            plt.hist(parameters[:, i], bins=bins, histtype='step')
            if use_mask:
                plt.hist(parameters[:, i][np.logical_not(mask)], bins=bins, histtype='step',color="green")
                plt.hist(parameters[:, i][mask], bins=bins, histtype='step',color="orange")
            ax.xaxis.tick_top()
            if i==co2_pos:
                ax.set_xlim(375,390)
            if i>x_start:
                ax.set_yticks([])
            else:
                plt.ylabel(f"Prior dist", fontsize = 23)
                #ax.yaxis.label.set_fontsize(23)

    offset = -y_start
    if show_distribution:
        offset = 1-y_start

    for i in range(x_start,x_end):
        for j in range(y_start,y_end):
            print(i,j,x_start,x_end,y_start,y_end,x_end-x_start+1, y_end-y_start, i-x_start+1+(x_end-x_start)*(j-y_start+1))
            ax=plt.subplot(y_end+offset, x_end-x_start, i-x_start+1+(x_end-x_start)*(j+offset))
            plt.subplots_adjust(hspace = .001, wspace=.001)
            if use_mask:
                plt.scatter(parameters[:, i][np.logical_not(mask)],parameters[:, j][np.logical_not(mask)],s=0.1,alpha=0.1)
                plt.scatter(parameters[:, i][mask],parameters[:, j][mask],s=0.1,alpha=0.1,color="orange")
            else:
                plt.scatter(parameters[:, i],parameters[:, j],s=0.1,alpha=0.1)
     
            if i==co2_pos:
                ax.set_xlim(375,390)
            if j==co2_pos:
                ax.set_ylim(375,390)
            if j<x_end-x_start-1:
                ax.set_xticks([])
            if i>x_start:
                ax.set_yticks([])
            else:
                if len(element_names[j])>5:
                    plt.ylabel(f"{element_names[j][:4]}..{element_names[j][-2:]}")
                else:
                    plt.ylabel(f"{element_names[j]}")
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                 #+ ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(23)
        plt.xlabel(element_names[i])


def show_elementcorrelations(parameters=output_params,n_parameters=n_parameters,offset=offset,name="",element_names=element_names):
    """Creates nice correlation plot of the given input parameters
    It's useful to see, if certain parameter are independent of each other of if using less input parameter doesn't change the impact on the problem.
    
    Keyword Arguments:
        parameters {[type]} -- [the array of parameter] (default: {orig_parameters})
        n_parameters {[int]} -- [number of compared parameter] (default: {n_parameters})
        offset {[int]} -- [number of first compared parameter in parameters] (default: {offset})
        name {str} -- [name, to show in plot] (default: {""})
        element_names {[str]} -- [names of parameters] (default: {element_names})
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
        bins=np.linspace(np.min(parameters[:, i+offset]),np.max(parameters[:, i+offset]),100)
        plt.hist(parameters[:, i+offset], bins=bins, histtype='step')
        if use_mask:
            plt.hist(parameters[:, i+offset][np.logical_not(mask)], bins=bins, histtype='step',color="green")
            plt.hist(parameters[:, i+offset][mask], bins=bins, histtype='step',color="orange")
        ax.xaxis.tick_top()
        if i==co2_pos:
            ax.set_xlim(375,390)
        if i>0:
            ax.set_yticks([])
        else:
            plt.ylabel(f"Prior dist", fontsize = 23)
            #ax.yaxis.label.set_fontsize(23)

    for i in range(n_parameters):
        for j in range(n_parameters):
            ax=plt.subplot(n_parameters+1, n_parameters, i+1+n_parameters*(j+1))
            plt.subplots_adjust(hspace = .001, wspace=.001)
            if use_mask:
                plt.scatter(parameters[:, i+offset][np.logical_not(mask)],parameters[:, j+offset][np.logical_not(mask)],s=0.1,alpha=0.1)
                plt.scatter(parameters[:, i+offset][mask],parameters[:, j+offset][mask],s=0.1,alpha=0.1,color="orange")
            else:
                plt.scatter(parameters[:, i+offset],parameters[:, j+offset],s=0.1,alpha=0.1)
     
            if i==co2_pos:
                ax.set_xlim(375,390)
            if j==co2_pos:
                ax.set_ylim(375,390)
            if j<n_parameters-1:
                ax.set_xticks([])
            if i>0:
                ax.set_yticks([])
            else:
                if len(element_names[j])>5:
                    plt.ylabel(f"{element_names[j][:4]}..{element_names[j][-2:]}")
                else:
                    plt.ylabel(f"{element_names[j]}")
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
                 #+ ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(23)
        plt.xlabel(element_names[i])
    

def plot_quality_map(value,name="Satelite positon",title="Width of uncertanty, depending on position"):
    """AKA world map

    
    Arguments:
        uncert_intervals {[type]} -- [description]
    
    Keyword Arguments:
        name {str} -- [description] (default: {"Satelite positon"})
        title {str} -- [description] (default: {"Width of uncertanty, depending on position"})
    """
    import geopandas
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.colors

    external_results=prepare_data.co2_ret_params[:,-4:]

    #co2_mask=np.logical_and(np.greater_equal(error,377),np.less(error,389))
    month_mask1 = np.isclose(external_results[:,0],1,1e-3)
    month_mask4 = np.isclose(external_results[:,0],4,1e-3)
    month_mask7 = np.isclose(external_results[:,0],7,1e-3)
    month_mask10 = np.isclose(external_results[:,0],10,1e-3)
    month_masks = [month_mask1,month_mask4,month_mask7,month_mask10]
    month = ["January","April","July","October"]
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    #fig = plt.figure(name,figsize=(20,15))
    #plt.tight_layout()
    #plt.title(title)

    fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(20,10))
    fig.tight_layout()
    #fig.set_title(title)
    levels = np.percentile(value, np.linspace(0,100,101))
    norm = matplotlib.colors.BoundaryNorm(levels,256)
    for i,ax in enumerate(axes.flat):
        #ax = plt.subplot(2, 2, 1)
        ax.set_aspect('equal')
        ax.set_title(f"{month[i]}")
        world.plot(ax=ax, color='lightblue', edgecolor='black')
        im = ax.scatter(external_results[:, 2][month_masks[i]],external_results[:,1][month_masks[i]],c=value[month_masks[i]],s=10,norm=norm)#,cmap=plt.get_cmap("plasma")
        #plt.colorbar(im)
        #plt.legend(title="in ppm")
        if i > 1:
            ax.set_xlabel("Longitude")
        else:
            ax.set_xticks([])
        if i%2 is 0:
            ax.set_ylabel("Latitude")   
        else:
            ax.set_yticks([])
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(23)

        print(month[i],value[month_masks[i]].max(),value[month_masks[i]].min())

    #divider = make_axes_locatable(fig)
    #cax = divider.append_axes("right", size="1%", pad=0.08)
    print("\n")
    ticks = np.percentile(value, np.linspace(0,100,9))
    print(value,value.shape,value.max(),value.min(),ticks)
    #cbar  = plt.colorbar(im, ax=axes.ravel().tolist(),ticks=ticks,aspect=20)#cax=cax

    plt.tight_layout()
    fig.subplots_adjust(right=0.89)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax,ticks=ticks)
    cbar.set_label("in ppm")

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label, cbar.ax.yaxis.label] +
        ax.get_xticklabels() + ax.get_yticklabels()+cbar.ax.get_yticklabels()):
            item.set_fontsize(23)
    #
    
    #fig.colorbar(im, ax=axes.ravel().tolist())

def plot():
    #for i in range(len(output_params[0])):
    #    plot_quality_map(output_params[:,i])

    
    plot_quality_map(output_params[:,co2_pos])
    #show_param_hists(output_x)
    ##show_param_hists(output_params,name="orig")
    #show_elementcorrelations()
    #show_nice_elementcorrelations()
    #show_nice_elementcorrelations(x_start=4,x_end= 8, y_start = 8, y_end = 11,name="mid_temp")
    #show_nice_elementcorrelations(x_start=0,x_end= 4, y_start = 2, y_end = 5,name="low_temp")
    #show_nice_elementcorrelations(x_start=13,x_end= 18, y_start = 14, y_end = 18,name="high_params",show_distribution=True)
    #show_one_input_data()
    ##show_spectra()
    #show_param_hists(new_params,n_parameters=5,name="0")


if __name__ == "__main__":
    plot()
    plt.show(block=False)
    input("Press enter to end!")