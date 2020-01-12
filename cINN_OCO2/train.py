
import torch
import torch.nn
import torch.optim
from torch.nn.functional import avg_pool2d, interpolate
from torch.autograd import Variable
import numpy as np

import config as c
print("train run_name",c.run_name)
import nets
import viz
import tqdm

import time

model = c.model
import help_train as ht


print("len of train_loader:",len(c.train_ana_loader))




def testINN(i_epoch):
    test_loss=[]
    print("interesting observations")
    print("FN_model_train_mode before test call",nets.model.training)
    print("INN_model_train_mode before test call",model.training)
    model.eval()
    print("FN_model_train_mode during test call",nets.model.training)
    print("INN_model_train_mode during test call",model.training)
    print("\n\n\n")
    for x_test,y_test, ana in c.test_ana_loader:
        x_test, y_test = x_test.to(c.device), y_test.to(c.device)
        with torch.no_grad():
            output, jac = model.forward(x_test, y_test)


            zz = torch.sum(output**2, dim=1)

            neg_log_likeli = 0.5 * zz - jac
            test_loss.append(torch.mean(neg_log_likeli).item())

    test_loss=np.mean(np.array(test_loss), axis=0)
    ht.sample_posterior(y_test,x_test, name = "test")

    model.train()
    print(f"Test Loss {i_epoch}: {test_loss}")
    print("\nTest loss_train: ",test_loss, f"{time.perf_counter() - t:.2f}s")
    
    viz.show_loss(test_loss,"test")


def trainINN(i_epoch):
    loss_history = []
    data_iter = iter(c.train_ana_loader)

    if i_epoch < 0:
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = c.lr_init * c.lr_reduce_factor
    if i_epoch == 0:
        for param_group in model.optimizer.param_groups:
            param_group['lr'] = c.lr_init

    #print("FN_model_train_mode before train call",nets.model.training)
    print("INN_model_train_mode before train call",model.training)
    print(f"Learningrate for epoch {i_epoch} is {model.scheduler.get_lr()[0]:.3e}")            

    for param_group in model.optimizer.param_groups:
            print(f"Start Learningrate for epoch {i_epoch} is {param_group['lr']:.3e}")
    

    iterator = tqdm.tqdm(enumerate(data_iter),
                        total=min(len(c.train_ana_loader), c.n_its_per_epoch),
                        leave=False,
                        mininterval=1.,
                        disable=(not c.progress_bar),
                        ncols=83)



    model.train()
    for i_batch , (x,cond,_) in iterator:
        cond, x = cond.to(c.device), x.to(c.device)
        model.optimizer.zero_grad()
        if c.do_rev:
            #this condition hasn't been updated for a while. Don't expect this to work
            def sample_outputs(sigma, out_shape, batchsize=4):
                return [sigma * torch.cuda.FloatTensor(torch.Size((batchsize, o))).normal_() for o in out_shape]
            
            z = sample_outputs(1., model.output_dimensions, c.batch_size)
            features = nets.model.features(cond)
            output = model.model(z, features, rev = True)

            #x_gen = model.combined_model.module.reverse_sample(z, cond.cuda())
            jac = model.model.log_jacobian(run_forward=False)
            l = 3.5 * torch.mean((x - output)**2) - torch.mean(jac)#/tot_output_size
        else:
            #default case
            z, jac = model.forward(x, cond)
            zz = torch.sum(z**2, dim=1)
            neg_log_likeli = 0.5 * zz - jac     
            l = torch.mean(neg_log_likeli) #/ tot_output_size

        l.backward()

        model.optim_step()
        loss_history.append([l.item()])
        assert not np.isnan(np.sum(np.array(l.item()))),f"\n loss_history {loss_history}"
        if i_batch+1 >= c.n_its_per_epoch:
            # somehow the data loader workers don't shut down automatically
            try:
                data_iter._shutdown_workers()
            except:
                pass

            iterator.close()
            break
    print(loss_history)
    
    ht.sample_posterior(cond,x, "train")

    epoch_losses = np.mean(np.array(loss_history), axis=0)

    print("Train loss",epoch_losses[0])
    print(epoch_losses.shape)
    print(epoch_losses)

    assert not np.isnan(np.sum(epoch_losses)),loss_history
    viz.show_loss(epoch_losses[0],"train")
    viz.show_hist(z.cpu().detach().numpy())



def train():
        
    if c.load_file:
        model.load(c.load_file)

    fn_weights = np.array(nets.read_params()[0])
    INN_weights = np.array(model.read_params()[0])
    print("fn weights init",np.sum(np.abs(fn_weights)))
    print("INN weights init",np.sum(np.abs(INN_weights)))
    #try:
    for i_epoch in range(-c.pre_low_lr, c.n_epochs):

        trainINN(i_epoch)
        
        old_fn_weights = fn_weights
        old_INN_weights = INN_weights
        fn_weights = np.array(nets.read_params()[0])
        INN_weights = np.array(model.read_params()[0])
        print("fn weights updated",np.sum(np.abs(fn_weights - old_fn_weights)))
        print("INN weights updated",np.sum(np.abs(INN_weights - old_INN_weights)))

        testINN(i_epoch)

        model.scheduler_step()

        if i_epoch % 4 == 0:
            ht.show_year_error()
        viz.make_step()
        if i_epoch > 0 and (i_epoch % c.checkpoint_save_interval) == 0:
            model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))

    model.save(c.filename)
    viz.close()

    import nice_eval\
    #alternative
    #evaluate.main()


def main():
    pass


t = time.perf_counter()
if __name__ == "__main__":

    train()
