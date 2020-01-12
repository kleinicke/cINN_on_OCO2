import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torchvision import datasets, transforms
import numpy as np
import copy

import config as c
import data.dataloader as dataloader
#import help_train as ht

import data.prepare_data as prepare_data

import time
import nets
#import viz
c.fn_pretrain_percentage = 3
c.fn_pretrain_log_interval = 10
c.predict_uncert = False
c.fn_pretrain_number_of_tests = 1

#model = nets.model
name=c.feature_net_name

if name == "fully_connected":
    model = nets.FullyConnected().to(c.device)
elif name == "cnn_long_solve":
    model = nets.Cnn_long_solve().to(c.device)#Cnn_long_solve().to(c.device)
elif name == "cnn_sep_solve":
    model = nets.Cnn_sep_solve().to(c.device)#Cnn_long_solve().to(c.device)
elif name == "Easy_solve":
    model = nets.Easy_solve().to(c.device)#Cnn_long_solve().to(c.device)
elif name == "Harder_solve":
    model = nets.Harder_solve().to(c.device)
elif name == "Cnn_add_sep_solve":
    model = nets.Cnn_add_sep_solve().to(c.device)
elif name == "seperate_hard_solve":
    model = nets.seperate_hard_solve().to(c.device)
elif name == "Cnn_long_combined_solve":
    model = nets.Cnn_long_combined_solve().to(c.device)
#prior_net = nets.PriorNet().to(c.device)

#test_loader, train_loader = dataloader.get_loaders(c.batch_size)#
test_loader, train_loader = c.test_ana_loader, c.train_ana_loader
print("len of train_loader:",len(train_loader))


#optim = torch.optim.SGD(model.parameters(), lr=c.fn_pretrain_lr, momentum=0.5)
#optimizer = optim.SGD(model.parameters(), lr=0.5e-2, momentum=0.5)
#scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.01**(1./c.fn_pretrain_percentage/2), last_epoch=-1)
params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
gamma = (c.decay_by)**(1./(c.fn_pretrain_percentage))
optim = torch.optim.Adam(params_trainable, lr=c.lr_init, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)

"""
class show_year_error():

    def __init__(self,network = "INN"):
        self.network = network
    
        #import data.data_helpers as data_helpers
        #self.years = ["2014", "2015", "2016", "2017", "2018"]
        #self.set_2014, self.set_2015, self.set_2016, self.set_2017, self.set_2018 = None, None, None, None, None
        #self.year_sets = {}#[None, None, None, None, None]
        #self.viz = viz
        #for i,loadset in enumerate(dataloader.loader):
        #    year = self.years[i]
        #    x,y,_ = data_helpers.concatenate_set(loadset,5000)
        #    self.year_sets[year] = (x,y)

    def next(self, model):
        model.eval()
        for year in c.dc.viz_years:
            #print(self.year_sets)
            x,y = c.year_sets[year]#self.year_sets[year]
            y_test, x_test = y.to(c.device), x#.to(c.device)
            errors = ht.show_error(name = year)#, mode = year)
            with torch.no_grad():
                #for i in range(10):
                test_output = model.fn_func(y_test)
                if self.network == "INN":
                    errors.add_error(test_output,x_test)
                else:
                    #print(test_output.shape[0],x_test.shape[0],"\n\n\n\n\n")
                    for i in range(test_output.shape[0]):
                        #print(test_output[i],x_test[i])
                        errors.add_error(test_output[i][None,:],x_test[i][None,:])

            errors.print_error()

        model.train()       
"""
def test_solver():
    model.eval()
    test_loss = 0
    test_losses = np.zeros(c.x_dim)
    #errors = ht.show_error("test")#, mode = "test")
    with torch.no_grad():
        #error_av = []
        #mean_er_av = []
        #mean_offset_av = []
        for x_test, y_test, ana_test in test_loader:

            y_test, x_test = y_test.to(c.device), x_test.to(c.device)
            test_output = model.fn_func(y_test)#forward_solve(y_test)

            if c.predict_uncert:
                xi = test_output[:,0]
                s = test_output[:,1]
                loss_a = (xi-x_test[:,0])**2/(2*s**2)+0.5*torch.log(s**2)
                loss = torch.mean(loss_a)

            else:
                loss_a = (test_output[:,:] - x_test[:,:])**2

                loss = torch.mean(loss_a)

            #test_loss += torch.mean(((test_output - x_test))**2).item() # sum up batch loss
            test_loss += loss.item() # sum up batch loss
            #test_losses += torch.mean(((test_output - x_test))**2, dim=0).detach().cpu().numpy()# sum up batch loss
            test_losses += loss.detach().cpu().numpy()# sum up batch loss
            #x_new=dataloader.prepare_data.x_to_params(x_test.detach().cpu().numpy())[0:512]
            #out_new=dataloader.prepare_data.x_to_params(test_output.detach().cpu().numpy())[0:512]
            #errors.add_error(test_output,x_test)
            #mean_er = torch.mean(torch.abs(test_output - x_test), dim = 0)
            #mean_offset = torch.mean((test_output - x_test), dim = 0)

            #mean_er_av.append(mean_er.detach().cpu().numpy())
            #mean_offset_av.append(mean_offset.detach().cpu().numpy())

            #error_av.append(np.average(np.abs(out_new-x_new),axis=0))

    #_,y,_ = next(iter(test_loader))
    #viz.show_graph(model,y_test,mode = "test")
    #print("absdiffs",mean_er.detach().cpu().numpy())
    #print("absdiffs",np.mean(np.array(mean_er_av),axis = 0))  
    #print("meandiffs",np.mean(np.array(mean_offset_av),axis = 0))
    #error_av = np.array(error_av)
    #x_new=dataloader.prepare_data.x_to_params(x_test.detach().cpu().numpy())[0]
    #out_new=dataloader.prepare_data.x_to_params(test_output.detach().cpu().numpy())[0]

    #x_new=dataloader.prepare_data.x_to_params(x_test.detach().cpu().numpy())[0:512]
    #out_new=dataloader.prepare_data.x_to_params(test_output.detach().cpu().numpy())[0:512]

    #print("\ndifference x to guessed for last batch:\n",np.average(np.abs(x_new-out_new),axis=0))
    #print("\nand for all batches:\n",np.average(error_av,axis=0))
    
    #errors.print_error()

    test_loss /= len(test_loader)
    test_losses /= len(test_loader)
    print(f'Testloss: {test_loss} each{test_losses}')
    #print(f'Test (eval) set: {test_losses}')
    model.train()
    return test_loss



def optim_step():
    optim.step()
    optim.zero_grad()

def scheduler_step():
    scheduler.step()
    
def train_solver(epoch, N_epochs,model=model,viz=False):
    model.train()
    start_t = time.perf_counter()
    #raw_index = c.ana_names.index("xco2_raw")
    #errors = ht.show_error("train")#, mode = "train")
    for batch_idx, (x, y, ana) in enumerate(train_loader):
        
        y, x = y.to(c.device), x.to(c.device)

        optim.zero_grad()
        model.zero_grad()

        output = model.fn_func(y)
        #print(output.shape,x.shape)
        if c.predict_uncert:
            xi = output[:,0]
            s = output[:,1]
            loss_a = (xi-x[:,0])**2/(2*s**2)+0.5*torch.log(s**2)
            loss = torch.mean(loss_a)
        else:
            loss_a = (output[:,:] - x[:,:])**2

            loss = torch.mean(loss_a)
        loss.backward()
        loss_a = torch.mean(loss_a, dim = 0)
        #print(loss_a)

        #errors.add_error(output,x)

        optim_step()

        #print trainloss
        if batch_idx % c.fn_pretrain_log_interval == 0:
            print(f'\rTrain Epoch: {epoch}/{N_epochs-1} [{batch_idx * len(x)}/{len(train_loader.dataset)}' 
                f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} Time: {time.perf_counter() - start_t:.1f}', end='')
        
        ##assert not torch.isnan(torch.sum(output)), (loss_a,x,y)

        #print testloss
        if (batch_idx) % int(len(train_loader.dataset)/len(x)/c.fn_pretrain_number_of_tests) == int(len(train_loader.dataset)/len(x)/c.fn_pretrain_number_of_tests)-1:
            print(f"\nTrain_loss {loss_a.detach().cpu().numpy()}")
            difference = prepare_data.x_to_params(output.detach().cpu().numpy()) - prepare_data.x_to_params(x.detach().cpu().numpy())
            tot_difference = np.mean(np.abs(difference), axis = 0)
            rel_difference = np.mean((difference), axis = 0)
            print(f"Train errors {tot_difference} and with mean at {rel_difference}")
            mean_er = torch.mean(torch.abs(output - x), dim = 0)
            print("mean error train",mean_er.detach().cpu().numpy()) 
            #errors.print_error()   
            break   
    return loss.item()


def save(name, model=model):
    model.to("cpu")
    torch.save({'opt':optim.state_dict(),
                'net':model.state_dict()}, name)
    print(f"saved fn model at {name}")
    model.to(c.device)


def load(name, load_opt = False, model=model):
    print("loads fn_model from ",name)
    state_dicts = torch.load(name,map_location='cpu')
    #print(state_dicts)
    model.load_state_dict(state_dicts['net'])
    model.to(c.device)
    if load_opt:
        try:
            optim.load_state_dict(state_dicts['opt'])
        except ValueError:
            print('Cannot load optimizer for some reason or other')

def read_params():
    params=[]
    names=[]
    for name,param in model.named_parameters():

        params.append(torch.sum(param.data).detach().cpu().numpy())
        names.append(name)
    return params, names

def train_fn():
    t = time.process_time()
    print(f"starts to train: {c.feature_net_file}")
    #load(c.feature_net_file+"_prior", model=prior_net)

    #optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.5)
    #optimizer = optim.SGD(model.parameters(), lr=0.5e-2, momentum=0.5)
    #params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    #feature_net.optim = torch.optim.Adam(feature_net.model.parameters(), lr=1e-3, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
    #_,y_test,_ = next(iter(c.test_ana_loader))
    #print(y_test.shape,y_test)
    #viz.show_graph(model,y_test,mode = "test")
    
    #year_error = show_year_error(network = "FN")
    for epoch in range(c.fn_pretrain_percentage):
        epoch_t = time.perf_counter()  
        print(f"\nTraining {epoch}\n")

        train_loss = train_solver(epoch, c.fn_pretrain_percentage)#
        print(f"\nTesting {epoch}\n")
        test_t = time.perf_counter()  
        test_loss = test_solver()
        delta_t = time.perf_counter() - test_t
        delta_t_epoch = time.perf_counter() - epoch_t
        print(f"Time for Epoch {epoch}: {delta_t_epoch:.3f}s and testing {delta_t:.3f}s")
        scheduler_step() 
        #optim.step()
        #viz.show_loss(train_loss,"train")
        #viz.show_loss(test_loss,"test")
        print("Model_mode",model.training)
        #year_error.next(model)
        #viz.make_step()
        
    print(f"Time to train solver in {c.fn_pretrain_percentage} iterations: {time.process_time() - t:.2f}s")


    save(c.feature_net_file)


    import feature_net_eval
"""
def train_prior( model=prior_net):
    t = time.process_time()
    print(f"starts with: {c.feature_net_file}")

    #optimizer = optim.SGD(model.parameters(), lr=5e-2, momentum=0.5)
    #optimizer = optim.SGD(model.parameters(), lr=0.5e-2, momentum=0.5)
    #params_trainable = list(filter(lambda p: p.requires_grad, model.parameters()))
    #feature_net.optim = torch.optim.Adam(feature_net.model.parameters(), lr=1e-3, betas=c.adam_betas, eps=1e-6, weight_decay=c.l2_weight_reg)
    #_,y_test,_ = next(iter(c.test_ana_loader))
    #print(y_test.shape,y_test)
    #viz.show_graph(model,y_test,mode = "test")
    
    #year_error = show_year_error(network = "FN")
    for epoch in range(c.fn_pretrain_percentage):
        epoch_t = time.perf_counter()  
        print(f"\nTraining {epoch}\n")

        train_loss = train_solver(epoch, c.fn_pretrain_percentage, model=model)#
        print(f"\nTesting {epoch}\n")
        test_t = time.perf_counter()  
        #test_loss = test_solver()
        delta_t = time.perf_counter() - test_t
        delta_t_epoch = time.perf_counter() - epoch_t
        print(f"Time for Epoch {epoch}: {delta_t_epoch:.3f}s and testing {delta_t:.3f}s")
        scheduler_step() 
        #optim.step()
        #viz.show_loss(train_loss,"train")
        #viz.show_loss(test_loss,"test")
        print("Model_mode",model.training)
        #if epoch%4==0:
            #year_error.next(model)
        #viz.make_step()
        
    print(f"Time to train solver in {c.fn_pretrain_percentage} iterations: {time.process_time() - t:.2f}s")


    save(c.feature_net_file+"_prior", model=model)


    #import feature_net_eval

"""
def load_fn():
    if c.additionally_trained_feature_net:
        print(f'loades fn: {c.feature_net_file+"_trained"}')
        #feature_net.model.load_state_dict(torch.load(c.feature_net_file+"_trained"))
        load(c.feature_net_file+"_trained")
    elif c.use_pretrained:
        print(f"loades fn: {c.feature_net_file}")
        load(c.feature_net_file)

        #model.load_state_dict(torch.load(c.feature_net_file))
    params, names = read_params()

if __name__ == "__main__":
    #train_prior()
    train_fn()