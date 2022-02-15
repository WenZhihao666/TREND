import torch
from torch import nn
from torch.nn import functional as F


class DGNN(nn.Module):
    def __init__(self, args):
        super(DGNN, self).__init__()
        self.args = args
        self.vars = nn.ParameterList()

        w = nn.Parameter(torch.ones(*[args.hid_dim, args.feat_dim]))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.hid_dim)))

        w = nn.Parameter(torch.ones(*[args.hid_dim, args.feat_dim]))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.hid_dim)))

        w = nn.Parameter(torch.ones(*[args.out_dim, args.hid_dim]))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.out_dim)))

        w = nn.Parameter(torch.ones(*[args.out_dim, args.hid_dim]))
        torch.nn.init.kaiming_normal_(w)
        self.vars.append(w)
        self.vars.append(nn.Parameter(torch.zeros(args.out_dim)))

        delta = nn.Parameter(torch.ones(*[1]))
        self.vars.append(delta)

    def forward(self, self_feat, one_hop_feat, two_hop_feat, e_time, his_time, his_his_time, neg=False):
        vars = self.vars

        x_s = F.linear(self_feat, vars[0], vars[1])
        x_n_one = F.linear(one_hop_feat, vars[2], vars[3])
        if neg == False:
            dif_t = e_time.reshape(-1, 1) - his_time
            decay = -vars[8] * dif_t
            soft_decay = F.softmax(decay, dim=1).reshape(-1, self.args.hist_len, 1)
            weighted_feat = soft_decay * (x_n_one.reshape(-1, self.args.hist_len, self.args.hid_dim))
            x_n_one = torch.sum(weighted_feat, dim=1)
        else:
            dif_t = e_time.reshape(-1, 1, 1) - his_time
            decay = -vars[8] * dif_t
            soft_decay = F.softmax(decay, dim=2).reshape(-1, self.args.neg_size, self.args.hist_len, 1)
            weighted_feat = soft_decay * (
                x_n_one.reshape(-1, self.args.neg_size, self.args.hist_len, self.args.hid_dim))
            x_n_one = torch.sum(weighted_feat, dim=2)
            x_s = x_s.reshape(-1, self.args.neg_size, self.args.hid_dim)
        x_s = x_s + x_n_one
        x_s_one = torch.relu(x_s)

        x_one_s = F.linear(one_hop_feat, vars[0], vars[1])
        x_n_two = F.linear(two_hop_feat, vars[2], vars[3])
        if neg == False:
            dif_t = his_time.reshape(-1, self.args.hist_len, 1) - his_his_time
            decay = -vars[8] * dif_t
            soft_decay = F.softmax(decay, dim=2).reshape(-1, self.args.hist_len, self.args.hist_len, 1)
            weighted_feat = soft_decay * (
                x_n_two.reshape(-1, self.args.hist_len, self.args.hist_len, self.args.hid_dim))
            x_n_two = torch.sum(weighted_feat, dim=2)
            x_one_s = x_one_s.reshape(-1, self.args.hist_len, self.args.hid_dim)
        else:
            dif_t = his_time.reshape(-1, self.args.neg_size, self.args.hist_len, 1) - his_his_time
            decay = -vars[8] * dif_t
            soft_decay = F.softmax(decay, dim=3).reshape(-1, self.args.neg_size, self.args.hist_len, self.args.hist_len,
                                                         1)
            weighted_feat = soft_decay * (
                x_n_two.reshape(-1, self.args.neg_size, self.args.hist_len, self.args.hist_len, self.args.hid_dim))
            x_n_two = torch.sum(weighted_feat, dim=3)
            x_one_s = x_one_s.reshape(-1, self.args.neg_size, self.args.hist_len, self.args.hid_dim)
        x_one_s = x_one_s + x_n_two
        x_one_s = torch.relu(x_one_s)

        x_s_one_final = F.linear(x_s_one, vars[4], vars[5])
        if neg == False:
            dif_t = e_time.reshape(-1, 1) - his_time
            decay = -vars[8] * dif_t
            soft_decay = F.softmax(decay, dim=1).reshape(-1, self.args.hist_len, 1)
            weighted_feat = soft_decay * x_one_s
            x_n_one_final = torch.sum(weighted_feat, dim=1)
        else:
            dif_t = e_time.reshape(-1, 1, 1) - his_time
            decay = -vars[8] * dif_t
            soft_decay = F.softmax(decay, dim=2).reshape(-1, self.args.neg_size, self.args.hist_len, 1)
            weighted_feat = soft_decay * x_one_s
            x_n_one_final = torch.sum(weighted_feat, dim=2)
        x_n_one_final = F.linear(x_n_one_final, vars[6], vars[7])
        x_s_final = x_s_one_final + x_n_one_final

        return x_s_final

    def parameters(self):
        return self.vars
