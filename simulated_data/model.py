import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable

import FrEIA.framework as Ff
import FrEIA.modules as Fm

import config as c
import nets


class CondNet(nn.Module):
    '''conditioning2 network'''
    #from experiments/colorization_minimal_example/model.py
    def __init__(self):
        super().__init__()


        self.subCondNets = nn.ModuleList([])
        self.bigCondNet = nets.model

        for _ in range(c.n_blocks):
            self.subCondNets.append(nn.Sequential(nn.Linear(256, 512),
                                    nn.ReLU(),nn.Dropout(p=c.fn_dropout),#nn.BatchNorm1d(512),
                                    nn.Linear(512, c.y_dim_features),
                                    nn.ReLU(),nn.Dropout(p=c.fn_dropout),
                                    #nn.LeakyReLU(),
                                    #nn.BatchNorm1d(c.y_dim_features),
                                    ))


    def forward(self, cond):
        c = self.bigCondNet.features(cond)
        outputs = []
        for m in self.subCondNets:
            outputs.append(m(c))
        return outputs


class CINN(nn.Module):
    '''cINN, including the ocnditioning network'''
    def __init__(self):
        super().__init__()

        self.cinn = self.build_inn()
        self.cond_net = CondNet()

        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        self.trainable_parameters += list(self.cond_net.parameters())
        for p in self.trainable_parameters:
            p.data = c.init_scale * torch.randn_like(p)

        gamma = (c.decay_by)**(1./(c.pre_low_lr+c.n_epochs))
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=c.lr_init, betas=c.adam_betas, weight_decay=c.l2_weight_reg)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=gamma)

    def build_inn(self):

        def fc_constr():
            return lambda ch_in, ch_out: nn.Sequential(nn.Linear(ch_in, c.internal_width), nn.ReLU(),nn.Dropout(p=c.fc_dropout),
                         nn.Linear(c.internal_width,  c.internal_width), nn.ReLU(),nn.Dropout(p=c.fc_dropout),
                         nn.Linear(c.internal_width,  ch_out))

        nodes = [Ff.InputNode(c.x_dim)]

        # outputs of the cond. net at different resolution levels
        conditions = []

        for i in range(c.n_blocks):
            conditions.append(Ff.ConditionNode(c.y_dim_features))

            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor':fc_constr(), 'clamp':c.exponent_clamping},
                                 conditions=conditions[i]))
            #nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed':i}, name=f'PERM_FC_{i}'))
            #nodes.append(Ff.Node(nodes[-1], Fm.ActNorm, {}, name=f'ActN{i}'))

        nodes.append(Ff.OutputNode(nodes[-1]))
        #torch.autograd.set_detect_anomaly(True)
        return Ff.ReversibleGraphNet(nodes + conditions, verbose=False)


    def forward(self,x, cond):
        z = self.cinn(x, c=self.cond_net(cond))
        jac = self.cinn.log_jacobian(run_forward=False)
        return z, jac

    def reverse_sample(self, z, y):
        x_samples = self.cinn(z, c=self.cond_net(y.to(c.device)), rev=True)
        return x_samples

    def optim_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def scheduler_step(self):
        self.scheduler.step()

    def read_params(self):
        params=[]
        names=[]
        for name,param in self.cinn.named_parameters():

            params.append(torch.sum(param.data).detach().cpu().numpy())
            names.append(name)
        return params, names

    def save(self,name):
        self.cinn.to("cpu")
        self.cond_net.to("cpu")
        torch.save({'opt':self.optimizer.state_dict(),
                    'net':self.cinn.state_dict(),
                    'cond_net':self.cond_net.state_dict(),
                    },name)
                    
        print(f"saved  model at {name}")
        self.cinn.to(c.device)
        self.cond_net.to(c.device)

    def load(self,name):
        print("loads model from ",name)
        state_dicts = torch.load(name,map_location='cpu')
        self.cinn.load_state_dict(state_dicts['net'])
        self.cond_net.load_state_dict(state_dicts['cond_net'])
        self.cinn.to(c.device)
        self.cond_net.to(c.device)
        
        try:
            pass
            #optim.load_state_dict(state_dicts['opt'])
        except ValueError:
            print('Cannot load optimizer for some reason or other')


model = CINN()