import torch
from torch import nn
from torch.nn import functional as F


class Scale_4(nn.Module):
    def __init__(self, args):
        super(Scale_4, self).__init__()
        self.vars = nn.ParameterList()
        self.args = args
        w1 = nn.Parameter(torch.ones(*[args.out_dim + 1, 2*args.out_dim]))
        torch.nn.init.kaiming_normal_(w1)
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(args.out_dim + 1)))

    def forward(self, x):
        vars = self.vars
        x = F.linear(x, vars[0], vars[1])
        # x = torch.relu(x)
        x = F.leaky_relu(x)
        # x = torch.squeeze(x)

        x = x.T
        x1 = x[:self.args.out_dim].T #.view(x.size(0), self.args.out_dim)
        x2 = x[self.args.out_dim:].T #.view(x.size(0), 1)
        para_list = [x1, x2]
        return para_list

    def parameters(self):
        return self.vars

class Shift_4(nn.Module):
    def __init__(self, args):
        super(Shift_4, self).__init__()
        self.args = args
        self.vars = nn.ParameterList()
        w1 = nn.Parameter(torch.ones(*[args.out_dim + 1, 2*args.out_dim]))
        torch.nn.init.kaiming_normal_(w1)
        self.vars.append(w1)
        self.vars.append(nn.Parameter(torch.zeros(args.out_dim + 1)))

    def forward(self, x):
        vars = self.vars
        x = F.linear(x, vars[0], vars[1])
        # x = torch.relu(x)
        x = F.leaky_relu(x)
        # x = torch.squeeze(x)

        x = x.T
        x1 = x[:self.args.out_dim].T #.view(x.size(0), self.args.out_dim)
        x2 = x[self.args.out_dim:].T #.view(x.size(0), 1)
        para_list = [x1, x2]
        return para_list

    def parameters(self):
        return self.vars
