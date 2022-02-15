
from __future__ import division
from torch.utils.data import Dataset
import numpy as np
import sys
import random
import copy
import torch
from sklearn import preprocessing

class DataHelper(Dataset):
    def __init__(self, file_path, node_feature_path, neg_size, hist_len, directed=False, transform=None, tlp_flag=False):
        self.node2hist = dict()
        self.neg_size = neg_size  # 5
        self.hist_len = hist_len  # 5
        self.directed = directed
        self.transform = transform

        # self.max_d_time = -sys.maxint  # Time interval [0, T]
        self.max_d_time = -sys.maxsize  # 1.0

        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)  # 10^8

        self.all_edge_index = []

        self.node_time_nodes = dict()
        self.node_set = set()
        self.degrees = dict()
        self.edge_list = []
        self.node_rate = {}
        self.edge_rate = {}
        self.node_sum = {}
        self.edge_sum = {}
        self.time_stamp = []
        self.time_edges_dict = {}
        self.time_nodes_dict = {}
        print('loading data...')
        infile = torch.load(file_path)
        for i in range(infile.size(0)):
            s_node = int(infile[i][0].item())  # source node
            t_node = int(infile[i][1].item())  # target node
            d_time = float(infile[i][2].item())  # time slot, delta t

            self.node_set.update([s_node, t_node])

            if s_node not in self.degrees:
                self.degrees[s_node] = 0
            if t_node not in self.degrees:
                self.degrees[t_node] = 0

            self.all_edge_index.append([s_node, t_node])

            if s_node not in self.node2hist:  # node2hist  {node: [(historical neighbor, time)], ……}
                self.node2hist[s_node] = list()
            if not directed:  # undirected
                if t_node not in self.node2hist:
                    self.node2hist[t_node] = list()

            if tlp_flag:
                if d_time >= 1.0:
                    continue
            # the new added node's degree is 0

            self.edge_list.append((s_node, t_node, d_time))  # edge list
            if not directed:
                self.edge_list.append((t_node, s_node, d_time))

            self.node2hist[s_node].append((t_node, d_time))
            if not directed:
                self.node2hist[t_node].append((s_node, d_time))  # because undirected, so add the inverse version

            if s_node not in self.node_time_nodes:
                self.node_time_nodes[s_node] = dict()  # for the new added s_node, create a dict for it
            if d_time not in self.node_time_nodes[s_node]:
                self.node_time_nodes[s_node][d_time] = list()  # for the new time,
            self.node_time_nodes[s_node][d_time].append(t_node)
            if not directed:  # undirected
                if t_node not in self.node_time_nodes:
                    self.node_time_nodes[t_node] = dict()
                if d_time not in self.node_time_nodes[t_node]:
                    self.node_time_nodes[t_node][d_time] = list()
                self.node_time_nodes[t_node][d_time].append(s_node)

            if d_time > self.max_d_time:
                self.max_d_time = d_time  # record the max time

            self.degrees[s_node] += 1  # node degree
            self.degrees[t_node] += 1


            self.time_stamp.append(d_time)
            if not self.time_edges_dict.__contains__(d_time):
                self.time_edges_dict[d_time] = []
            self.time_edges_dict[d_time].append((s_node, t_node))
            if not self.time_nodes_dict.__contains__(d_time):
                self.time_nodes_dict[d_time] = []
            self.time_nodes_dict[d_time].append(s_node)
            self.time_nodes_dict[d_time].append(t_node)

        # a = torch.load(node_feature_path).numpy()
        self.node_features = torch.load(node_feature_path).numpy()
        self.node_features = preprocessing.StandardScaler().fit_transform(self.node_features)

        # print("degree_features", degree_features[0:5])
        self.node_list = sorted(list(self.node_set))
        self.time_stamp = sorted(list(set(self.time_stamp)))  # !!! time from 0 to 1
        # print('time minimum:', min(self.time_stamp))
        # print('time maxmum:', max(self.time_stamp))

        self.node_dim = len(self.node_set)
        self.data_size = 0
        for s in self.node2hist:
            hist = self.node2hist[s]
            hist = sorted(hist,
                          key=lambda x: x[1])  # from past(0) to now(1). This supports the events ranked in time order.
            self.node2hist[s] = hist
            self.data_size += len(self.node2hist[s])

        self.max_nei_len = max(map(lambda x: len(x), self.node2hist.values()))  # 955
        # print('#nodes: {}, #edges: {}, # train time_stamp: {}'.
        #       format(self.node_dim, len(self.edge_list), len(self.time_stamp)))
        # print('avg. degree: {}'.format(sum(self.degrees.values()) / len(self.degrees)))
        # print('max neighbors length: {}'.format(self.max_nei_len))
        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for s_node in self.node2hist:
            for t_idx in range(len(self.node2hist[
                                       s_node])):  # Note the range here, which means from 0 to the number of historical neighbors
                self.idx2source_id[
                    idx] = s_node
                self.idx2target_id[
                    idx] = t_idx
                idx += 1


        print('init. neg_table...')
        self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table()


    def get_node_dim(self):
        return self.node_dim

    def get_max_d_time(self):
        return self.max_d_time

    def init_neg_table(self):
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        for k in range(self.node_dim):
            tot_sum += np.power(self.degrees[k],
                                self.NEG_SAMPLING_POWER)
        for k in range(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                cur_sum += np.power(self.degrees[n_id], self.NEG_SAMPLING_POWER)
                por = cur_sum / tot_sum
                n_id += 1
            self.neg_table[k] = n_id - 1  # negtive table size = 1e8, element inside range from 1~number of total nodes

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # sampling via htne
        s_node = self.idx2source_id[idx]
        t_idx = self.idx2target_id[idx]  # To ensure that s_node must
        t_node = self.node2hist[s_node][t_idx][0]  # get the global idx for the target node
        e_time = self.node2hist[s_node][t_idx][1]

        if t_idx == 0:
            s_his_nodes = np.array(s_node).repeat(self.hist_len)
            s_his_times = np.array(e_time).repeat(self.hist_len)
        else:
            if t_idx < self.hist_len:
                s_nei_idx = np.random.choice(len(self.node2hist[s_node][:t_idx]), self.hist_len, replace=True)
            else:
                s_nei_idx = np.random.choice(len(self.node2hist[s_node][:t_idx]), self.hist_len, replace=False)

            s_his = [(self.node2hist[s_node][:t_idx])[i] for i in s_nei_idx]
            s_his_nodes = [h[0] for h in s_his]
            s_his_times = np.array([h[1] for h in s_his])

        t_his_list = self.node2hist[t_node]
        s_idx = t_his_list.index((s_node, e_time))

        if s_idx == 0:
            t_his_nodes = np.array(t_node).repeat(self.hist_len)
            t_his_times = np.array(e_time).repeat(self.hist_len)
        else:
            if s_idx < self.hist_len:
                t_nei_idx = np.random.choice(len(t_his_list[:s_idx]), self.hist_len, replace=True)
            else:
                t_nei_idx = np.random.choice(len(t_his_list[:s_idx]), self.hist_len, replace=False)

            t_his = [(t_his_list[:s_idx])[i] for i in t_nei_idx]
            t_his_nodes = [h[0] for h in t_his]
            t_his_times = np.array([h[1] for h in t_his])

        s_his_tidx_list = []
        for i in range(len(s_his_nodes)):
            s_node_his = self.node2hist[s_his_nodes[i]]
            his_tidx = np.argwhere(np.array([list(i) for i in s_node_his])[:, 1] <= s_his_times[i])[-1].item()
            s_his_tidx_list.append(his_tidx+1)


        s_his_his_nodes_list = []
        s_his_his_times_list = []
        for i in range(len(s_his_nodes)):
            if s_his_tidx_list[i] < self.hist_len:
                s_his_nei_idx = np.random.choice(np.arange(
                    len(self.node2hist[s_his_nodes[i]][:s_his_tidx_list[i]])), self.hist_len, replace=True)
            else:
                s_his_nei_idx = np.random.choice(np.arange(len(self.node2hist[s_his_nodes[i]][:s_his_tidx_list[i]])), self.hist_len, replace=False)

            s_his_his = [(self.node2hist[s_his_nodes[i]][:s_his_tidx_list[i]])[a] for a in s_his_nei_idx]
            s_his_his_nodes = [h[0] for h in s_his_his]
            s_his_his_times = [h[1] for h in s_his_his]

            s_his_his_times_list.append(s_his_his_times)
            s_his_his_nodes_list.append(s_his_his_nodes)

        s_his_his_times_list = np.array(s_his_his_times_list)
        s_his_his_nodes_list = np.array(s_his_his_nodes_list).astype(int)

        t_his_tidx_list = []
        for i in range(len(t_his_nodes)):
            t_node_his = self.node2hist[t_his_nodes[i]]
            his_tidx = np.argwhere(np.array([list(i) for i in t_node_his])[:, 1] <= t_his_times[i])[-1].item()
            t_his_tidx_list.append(his_tidx+1)

        t_his_his_nodes_list = []
        t_his_his_times_list = []
        for i in range(len(t_his_nodes)):
            if t_his_tidx_list[i] < self.hist_len:
                t_his_nei_idx = np.random.choice(np.arange(
                    len(self.node2hist[t_his_nodes[i]][:t_his_tidx_list[i]])), self.hist_len, replace=True)

            else:
                t_his_nei_idx = np.random.choice(np.arange(
                    len(self.node2hist[t_his_nodes[i]][:t_his_tidx_list[i]])), self.hist_len, replace=False)

            t_his_his = [(self.node2hist[t_his_nodes[i]][:t_his_tidx_list[i]])[a] for a in t_his_nei_idx]

            t_his_his_nodes = [h[0] for h in t_his_his]
            t_his_his_times = [h[1] for h in t_his_his]

            t_his_his_times_list.append(t_his_his_times)
            t_his_his_nodes_list.append(t_his_his_nodes)

        t_his_his_times_list = np.array(t_his_his_times_list)
        t_his_his_nodes_list = np.array(t_his_his_nodes_list).astype(int)

        # negtive_sampling part
        neg_s_nodes = self.negative_sampling().astype(int)

        neg_tidx_list = []
        for i in range(self.neg_size):
            neg_node = self.node2hist[neg_s_nodes[i]]
            if neg_node == []:
                neg_tidx_list.append(0)
            elif np.array([list(i) for i in neg_node])[:, 1][0] > e_time:
                neg_tidx_list.append(0)
            else:
                his_tidx = np.argwhere(np.array([list(i) for i in neg_node])[:, 1] <= e_time)[-1].item()
                neg_tidx_list.append(his_tidx+1)

        neg_his_nodes_list = []
        neg_his_times_list = []
        # neg_his_len = []
        for i in range(self.neg_size):
            if len(self.node2hist[neg_s_nodes[i]][:neg_tidx_list[i]]) == 0:
                neg_his_nodes = neg_s_nodes[i].repeat(self.hist_len)
                neg_his_times = np.array(e_time).repeat(self.hist_len)
            else:
                if neg_tidx_list[i] < self.hist_len:
                        neg_nei_idx = np.random.choice(len(self.node2hist[neg_s_nodes[i]][:neg_tidx_list[i]]), self.hist_len, replace=True)
                else:
                    neg_nei_idx = np.random.choice(len(self.node2hist[neg_s_nodes[i]][:neg_tidx_list[i]]), self.hist_len, replace=False)

                neg_his = [(self.node2hist[neg_s_nodes[i]][:neg_tidx_list[i]])[a] for a in neg_nei_idx]
                neg_his_nodes = [h[0] for h in neg_his]
                neg_his_times = [h[1] for h in neg_his]

            neg_his_nodes_list.append(neg_his_nodes)
            neg_his_times_list.append(neg_his_times)
            # neg_his_len.append(len(neg_his))

        neg_his_nodes_list = np.array(neg_his_nodes_list).astype(int)
        neg_his_times_list = np.array(neg_his_times_list)

        neg_his_idx_list = []
        for i in range(self.neg_size):
            his_idx_list = []
            for j in range(self.hist_len):
                neg_node_his = self.node2hist[neg_his_nodes_list[i][j]]
                if neg_node_his ==[]:
                    his_idx = 0
                elif np.array([list(i) for i in neg_node_his])[:, 1][0] > e_time:
                    his_idx = 0
                else:
                    his_idx = np.argwhere(np.array([list(i) for i in neg_node_his])[:, 1] <= neg_his_times_list[i][j])[-1].item()
                his_idx_list.append(his_idx+1)
            neg_his_idx_list.append(his_idx_list)
        neg_his_idx_list = np.array(neg_his_idx_list)

        neg_his_his_nodes_list = []
        neg_his_his_times_list = []
        for i in range(self.neg_size):
            neg_h_h_nodes_list = []
            neg_h_h_times_list = []
            for j in range(self.hist_len):
                if len(self.node2hist[neg_his_nodes_list[i][j]][0:neg_his_idx_list[i][j]]) == 0:
                    neg_his_his_nodes = neg_his_nodes_list[i][j].repeat(self.hist_len)
                    neg_his_his_times = np.array(e_time).repeat(self.hist_len)
                else:
                    if neg_his_idx_list[i][j] < self.hist_len:
                        neg_his_his_idx = np.random.choice(len(self.node2hist[neg_his_nodes_list[i][j]][0:neg_his_idx_list[i][j]]), self.hist_len, replace=True)
                    else:
                        neg_his_his_idx = np.random.choice(len(self.node2hist[neg_his_nodes_list[i][j]][0:neg_his_idx_list[i][j]]), self.hist_len, replace=False)

                    neg_his_his = [(self.node2hist[neg_his_nodes_list[i][j]][0:neg_his_idx_list[i][j]])[a] for a in neg_his_his_idx]
                    neg_his_his_nodes = [h[0] for h in neg_his_his]
                    neg_his_his_times = [h[1] for h in neg_his_his]

                neg_h_h_nodes_list.append(neg_his_his_nodes)
                neg_h_h_times_list.append(neg_his_his_times)
            neg_h_h_nodes_list = np.array(neg_h_h_nodes_list)
            neg_h_h_times_list = np.array(neg_h_h_times_list)
            neg_his_his_nodes_list.append(neg_h_h_nodes_list)
            neg_his_his_times_list.append(neg_h_h_times_list)

        neg_his_his_nodes_list = np.array(neg_his_his_nodes_list).astype(int)
        neg_his_his_times_list = np.array(neg_his_his_times_list)

        s_self_feat = self.node_features[s_node]
        s_one_hop_feat = self.node_features[s_his_nodes]
        s_two_hop_feat = []
        for i in range(self.hist_len):
            s_two_feat = self.node_features[s_his_his_nodes_list[i]]
            s_two_hop_feat.append(s_two_feat)
        s_two_hop_feat = np.array(s_two_hop_feat)

        t_self_feat = self.node_features[t_node]
        t_one_hop_feat = self.node_features[t_his_nodes]
        t_two_hop_feat = []
        for i in range(self.hist_len):
            t_two_feat = self.node_features[t_his_his_nodes_list[i]]
            t_two_hop_feat.append(t_two_feat)
        t_two_hop_feat = np.array(t_two_hop_feat)

        neg_self_feat = self.node_features[neg_s_nodes]
        neg_one_hop_feat = []
        for i in range(self.neg_size):
            neg_one_feat = self.node_features[neg_his_nodes_list[i]]
            neg_one_hop_feat.append(neg_one_feat)
        neg_one_hop_feat = np.array(neg_one_hop_feat)

        neg_two_hop_feat = []
        for i in range(self.neg_size):
            neg_two_h_feat = []
            for j in range(self.hist_len):
                neg_t_h_f = self.node_features[neg_his_his_nodes_list[i][j]]
                neg_two_h_feat.append(neg_t_h_f)
            neg_two_h_feat = np.array(neg_two_h_feat)
            neg_two_hop_feat.append(neg_two_h_feat)
        neg_two_hop_feat = np.array(neg_two_hop_feat)

        s_edge_rate = len(self.node_time_nodes[s_node][e_time])

        # print('s_edge_rate', s_edge_rate)

        sample = {
            # 's_node': s_node,  # e.g., 5424
            # 't_node': t_node,  # e.g., 5427
            'event_time': e_time,
            's_history_times': s_his_times,
            't_history_times': t_his_times,
            's_his_his_times_list': s_his_his_times_list,
            't_his_his_nodes_list': t_his_his_nodes_list,
            't_his_his_times_list': t_his_his_times_list,
            's_self_feat': s_self_feat,
            's_one_hop_feat': s_one_hop_feat,
            's_two_hop_feat': s_two_hop_feat,
            't_self_feat': t_self_feat,
            't_one_hop_feat': t_one_hop_feat,
            't_two_hop_feat': t_two_hop_feat,
            'neg_his_times_list': neg_his_times_list,
            'neg_his_his_times_list': neg_his_his_times_list,
            'neg_self_feat': neg_self_feat,
            'neg_one_hop_feat': neg_one_hop_feat,
            'neg_two_hop_feat': neg_two_hop_feat,
            's_edge_rate': s_edge_rate,

        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def negative_sampling(self):
        rand_idx = np.random.randint(0, self.neg_table_size, (self.neg_size,))
        sampled_nodes = self.neg_table[rand_idx]
        return sampled_nodes
