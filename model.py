import torch
from torch import nn, optim
from dgnn import DGNN
from film import Scale_4, Shift_4
from Emlp import EMLP
from node_relu import Node_edge


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.l2reg = args.l2_reg
        self.ncoef = args.ncoef
        self.EMLP = EMLP(args)
        # self.grow_f = E_increase(args.edge_grow_input_dim)
        self.gnn = DGNN(args)
        self.scale_e = Scale_4(args)
        self.shift_e = Shift_4(args)
        self.node_edge = Node_edge(args)

        # self.g_optim = optim.Adam(self.grow_f.parameters(), lr=args.lr)

        self.optim = optim.Adam([{'params': self.gnn.parameters()},
                                 {'params': self.EMLP.parameters()},
                                 {'params': self.scale_e.parameters()},
                                 {'params': self.shift_e.parameters()},
                                 {'params': self.node_edge.parameters()},
                                 ], lr=args.lr)

    def forward(self, s_self_feat, s_one_hop_feat, s_two_hop_feat,
                t_self_feat, t_one_hop_feat, t_two_hop_feat,
                neg_self_feat, neg_one_hop_feat, neg_two_hop_feat,
                e_time, s_his_time, s_his_his_time,
                t_his_time, t_his_his_time,
                neg_his_time, neg_his_his_time,
                s_edge_rate,
                training=True):
        s_emb = self.gnn(s_self_feat, s_one_hop_feat, s_two_hop_feat, e_time, s_his_time, s_his_his_time)
        t_emb = self.gnn(t_self_feat, t_one_hop_feat, t_two_hop_feat, e_time, t_his_time, t_his_his_time)
        neg_embs = self.gnn(neg_self_feat, neg_one_hop_feat, neg_two_hop_feat, e_time, neg_his_time, neg_his_his_time,
                            neg=True)

        ij_cat = torch.cat((s_emb, t_emb), dim=1)
        alpha_ij = self.scale_e(ij_cat)
        beta_ij = self.shift_e(ij_cat)
        theta_e_new = []
        for s in range(2):
            theta_e_new.append(torch.mul(self.EMLP.parameters()[s], (alpha_ij[s] + 1)) + beta_ij[s])

        p_dif = (s_emb - t_emb).pow(2)
        p_scalar = (p_dif * theta_e_new[0]).sum(dim=1, keepdim=True)
        p_scalar += theta_e_new[1]
        p_scalar_list = p_scalar

        event_intensity = torch.sigmoid(p_scalar_list) + 1e-6
        log_event_intensity = torch.mean(-torch.log(event_intensity))

        dup_s_emb = s_emb.repeat(1, 1, self.args.neg_size)
        dup_s_emb = dup_s_emb.reshape(s_emb.size(0), self.args.neg_size, s_emb.size(1))

        neg_ij_cat = torch.cat((dup_s_emb, neg_embs), dim=2)
        neg_alpha_ij = self.scale_e(neg_ij_cat)
        neg_beta_ij = self.shift_e(neg_ij_cat)
        neg_theta_e_new = []
        for s in range(2):
            neg_theta_e_new.append(torch.mul(self.EMLP.parameters()[s], (neg_alpha_ij[s] + 1)) + neg_beta_ij[s])

        neg_dif = (dup_s_emb - neg_embs).pow(2)
        neg_scalar = (neg_dif * neg_theta_e_new[0]).sum(dim=2, keepdim=True)
        neg_scalar += neg_theta_e_new[1]
        big_neg_scalar_list = neg_scalar

        neg_event_intensity = torch.sigmoid(- big_neg_scalar_list) + 1e-6

        neg_mean_intensity = torch.mean(-torch.log(neg_event_intensity))

        pos_l2_loss = [torch.norm(s, dim=1) for s in alpha_ij]
        pos_l2_loss = [torch.mean(s) for s in pos_l2_loss]
        pos_l2_loss = torch.sum(torch.stack(pos_l2_loss))
        pos_l2_loss += torch.sum(torch.stack([torch.mean(torch.norm(s, dim=1)) for s in beta_ij]))
        neg_l2_loss = torch.sum(torch.stack([torch.mean(torch.norm(s, dim=2)) for s in neg_alpha_ij]))
        neg_l2_loss += torch.sum(torch.stack([torch.mean(torch.norm(s, dim=2)) for s in neg_beta_ij]))

        l2_loss = pos_l2_loss + neg_l2_loss
        l2_loss = l2_loss * self.l2reg

        delta_e = self.node_edge(s_emb)
        node_loss = nn.SmoothL1Loss()
        l_node = node_loss(delta_e, s_edge_rate.reshape(s_edge_rate.size(0), 1))
        # l_node = torch.sqrt(l_node)
        l_node = self.ncoef * l_node

        L = log_event_intensity + neg_mean_intensity + l2_loss + l_node

        if training == True:
            self.optim.zero_grad()
            L.backward()
            self.optim.step()

        return round((L.detach().clone()).cpu().item(),
                     4), s_emb.detach().clone().cpu(), t_emb.detach().clone().cpu(), dup_s_emb.detach().clone().cpu(), neg_embs.detach().clone().cpu()
