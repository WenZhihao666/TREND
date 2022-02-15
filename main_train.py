import sys

import os.path as osp
import time
import torch
import numpy as np
import random
import math
import time
import argparse
from data_dyn_cite import DataHelper
from torch.utils.data import DataLoader
from model import Model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
FType = torch.FloatTensor
LType = torch.LongTensor


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    setup_seed(args.seed)
    Data = DataHelper(args.file_path, args.node_feature_path, args.neg_size, args.hist_len, args.directed,
                      tlp_flag=args.tlp_flag)

    model = Model(args).to(device)
    model.train()
    for j in range(args.epoch_num):
        loader = DataLoader(Data, batch_size=args.batch_size, shuffle=True, num_workers=5)
        for i_batch, sample_batched in enumerate(loader):
            loss, _, _, _, _, = model.forward(
                sample_batched['s_self_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['s_one_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['s_two_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),

                sample_batched['t_self_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['t_one_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['t_two_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),

                sample_batched['neg_self_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['neg_one_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),
                sample_batched['neg_two_hop_feat'].type(FType).reshape(-1, args.feat_dim).to_sparse().to(device),

                sample_batched['event_time'].type(FType).to(device),
                sample_batched['s_history_times'].type(FType).to(device),
                sample_batched['s_his_his_times_list'].type(FType).to(device),
                sample_batched['t_history_times'].type(FType).to(device),
                sample_batched['t_his_his_times_list'].type(FType).to(device),
                sample_batched['neg_his_times_list'].type(FType).to(device),
                sample_batched['neg_his_his_times_list'].type(FType).to(device),
                sample_batched['s_edge_rate'].type(FType).to(device),
            )
            if j == 0:
                if i_batch % 10 == 0:
                    print('batch_{} event_loss:'.format(i_batch), loss)

        print('ep_{}_event_loss:'.format(j + 1), loss)

    torch.save(model.state_dict(), '../res/cite/model.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', type=str, default='./data/cite/emb_edges.pt')
    parser.add_argument('--node_feature_path', type=str, default='./data/cite/sorted_emb_feat.pt')
    parser.add_argument('--neg_size', type=int, default=1)
    parser.add_argument('--hist_len', type=int, default=10)
    parser.add_argument('--directed', type=bool, default=False)
    parser.add_argument('--epoch_num', type=int, default=20, help='epoch number')
    parser.add_argument('--tlp_flag', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hid_dim', type=int, default=16)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--out_dim', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ncoef', type=float, default=0.01)
    parser.add_argument('--l2_reg', type=float, default=0.001)

    args = parser.parse_args()

    main(args)
