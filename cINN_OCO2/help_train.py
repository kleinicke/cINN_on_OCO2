"""Contains some functions for evaluation of the training process
"""

import numpy as np
import data.dataloader as dataloader
import data.prepare_data as prepare_data
import torch
import config as c

model = c.model#0


def set_Model(newModel):
    """pass over model, when model is not the standard INN.

    Arguments:
        newModel {Model} -- Model that is currently used
    """
    global model
    model = newModel
    print("model is set")
    print(model)

#using this function changes training in some cases...
def sample_posterior(y_it, x_it, name = "Show",  N=128, mode = False, model = model):
    errors = show_error(name = name)#, mode = mode)
    z = torch.randn(N, c.x_dim).to(c.device)

    model.eval()

    for idx, y in enumerate(y_it):

        y = y.expand(N,-1)
        with torch.no_grad():
            x_samples = model.reverse_sample(z,y)
            
        errors.add_error(x_samples,x_it[idx].to(c.device))

    errors.print_error()
    model.train()



class show_error():
    def __init__(self, name = "test", mode = False, network = "INN", visualize = True, loss_fnc = lambda x,y: x-y):
        self.error_av = []
        self.mean_abs_ers = []
        self.median_abs_ers = []
        self.mean_x_offsets = []
        self.median_x_offset = []
        self.name = name
        self.mode = mode
        self.network = network #INN or FN
        self.visualize = visualize
        self.l_f = loss_fnc
        self.iterations = 0

    def add_error(self,output, x_gt):

        output = output.detach().cpu().numpy()
        x_gt = x_gt.detach().cpu().numpy()
        x_new=dataloader.prepare_data.x_to_params(x_gt)
        out_new=dataloader.prepare_data.x_to_params(output)
        #print(out_new.shape, x_new.shape)
        if self.network == "INN":
            mean_abs_er = np.mean(np.abs(self.l_f(output, x_gt)), axis = 0)
            median_abs_er = np.median(np.abs(self.l_f(output, x_gt)), axis = 0)
            mean_x_offset = np.mean(self.l_f(output, x_gt), axis = 0)
            median_x_offset = np.median(self.l_f(output, x_gt), axis = 0)
            self.error_av.append(np.average(np.abs(self.l_f(out_new, x_new)),axis=0))

        else:
            mean_abs_er = np.abs(self.l_f(output, x_gt))
            median_abs_er = np.abs(self.l_f(output, x_gt))
            mean_x_offset = np.array(self.l_f(output, x_gt))
            median_x_offset = np.array(self.l_f(output, x_gt))
            self.error_av.append(np.abs(self.l_f(out_new, x_new)))

        self.mean_abs_ers.append(mean_abs_er)
        self.median_abs_ers.append(median_abs_er)
        mean_x_offset = np.dot(mean_x_offset, np.linalg.inv(prepare_data.w_x))
        median_x_offset = np.dot(median_x_offset, np.linalg.inv(prepare_data.w_x))
        self.mean_x_offsets.append(mean_x_offset)
        self.median_x_offset.append(median_x_offset)


        self.error_av.append(np.average(np.abs(out_new-x_new),axis=0))
        self.iterations += 1
    
    def reset(self):
        self.error_av = []
        self.mean_abs_ers = []
        self.median_abs_ers = []
        self.mean_x_offsets = []
        self.median_x_offset = []
        self.iterations = 0


    def print_error(self):
        if self.visualize:
            import viz


        if self.visualize:
            if self.network == "INN":
                #print("correct branch")
                meanabdiffs = np.mean(np.array(self.mean_abs_ers),axis = 0)
                meadabdiffs = np.mean(np.array(self.median_abs_ers),axis = 0)
                offsets_den = np.mean(np.array(self.mean_x_offsets),axis = 0)
                offsets_median_den = np.mean(np.array(self.median_x_offset),axis = 0)
                param_differences = np.average(np.array(self.error_av),axis=0)
            else:
                meanabdiffs = np.mean(np.array(self.mean_abs_ers[0]),axis = 0)
                meadabdiffs = np.mean(np.array(self.median_abs_ers[0]),axis = 0)
                offsets_den = np.mean(np.array(self.mean_x_offsets[0]),axis = 0)
                offsets_median_den = np.mean(np.array(self.median_x_offset[0]),axis = 0)
                param_differences = np.average(np.array(self.error_av[0]),axis=0)
            

            viz.show_stats([meanabdiffs[0],meadabdiffs[0],offsets_den[0],offsets_median_den[0],param_differences[0]], self.name)
            
            offsets_den_std = np.std(np.array(self.mean_x_offsets),axis = 0)
            print(np.array(self.mean_x_offsets).shape)#, self.mean_x_offsets)
            param_differences_std = np.std(np.array(self.error_av),axis=0)

            viz.show_std(offsets_den_std[0], mode = self.name, name = "offsets_den/std")
            viz.show_std(param_differences_std[0], mode = self.name, name = "param_differences/std")
            if self.network == "INN":
                bootstrap_offset = []
                bootstrap_error = [] 
                #print("bootshape 1",np.array(self.mean_x_offsets).shape)
                for i in range(5):
                    bs_offsets_den = np.mean(np.array(self.mean_x_offsets)[i::5],axis = 0)
                    #print("bootshape 2",bs_offsets_den.shape,np.array(self.mean_x_offsets)[i::5].shape)
                    bs_param_differences = np.average(np.array(self.error_av)[i::5],axis=0)
                    bootstrap_offset.append(bs_offsets_den[0])
                    bootstrap_error.append(bs_param_differences[0])
                
                bootstrap_offset = np.array(bootstrap_offset)
                print("bootstrap_offset",bootstrap_offset,np.mean(np.square(bootstrap_offset)))
                print("with iterations",self.iterations,5/self.iterations)
                bootstrap_error = np.array(bootstrap_error)
                #print("bootshape 3",bootstrap_offset.shape)
                #print("min max",bootstrap_offset.min(),bootstrap_offset.max())
                bootstrap_offset_max = max(abs(offsets_den[0]-bootstrap_offset.max()),abs(offsets_den[0]-bootstrap_offset.min()))
                bootstrap_error_max = max(abs(param_differences[0]-bootstrap_error.max()),abs(param_differences[0]-bootstrap_error.min()))
                viz.show_std(bootstrap_offset_max, mode = self.name, name = "offsets_den/bs")
                viz.show_std(bootstrap_error_max, mode = self.name, name = "param_differences/bs")
                viz.show_std(max(abs(bootstrap_offset.max()),abs(bootstrap_offset.min())), mode = self.name, name = "offsets_den/bs_max")
                viz.show_std(max(abs(bootstrap_error.max()),abs(bootstrap_error.min())), mode = self.name, name = "param_differences/bs_max")
                viz.show_std((bootstrap_offset.max()), mode = self.name, name = "offsets_den/bs_max_max")
                viz.show_std(((bootstrap_offset.min())), mode = self.name, name = "offsets_den/bs_max_min")
                viz.show_std(np.mean(np.square(bootstrap_offset))*self.iterations/5/offsets_den_std[0], mode = self.name, name = "offsets_den/E(bias5^2)*iterations/5")

def show_year_error():
    model.eval()
    for year in c.dc.viz_years:
        x,y = c.year_sets[year]
        y_test, x_test = y.to(c.device), x.to(c.device)
        with torch.no_grad():
            sample_posterior(y_test,x_test, year)
           
    model.train()
