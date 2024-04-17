import os

os.environ['DGLBACKEND'] = 'pytorch'
import torch
import pandas as pd
import dgl
from dgl.data import DGLDataset
from dgl.data import Subset
from random import shuffle
from dgl import RemoveSelfLoop
from random import sample
from dgl import transforms as T
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from torch_geometric.utils import coalesce, scatter, softmax
from dgl import ToSimple
import copy
import torch.nn as nn
from imports import device
import dgl.function as fn
import numpy as np
import torch as th
from dgl.nn.pytorch import GINConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import TAGConv

class pre_embedding(nn.Module):

    def __init__(self, in_dim, out_dim, n_class, gnn_type):
        super(pre_embedding, self).__init__()

        self.pred_lin = nn.Sequential(nn.Linear(out_dim, int(out_dim / 2)), nn.ReLU(),
                                      nn.Linear(int(out_dim / 2), n_class))

        self.transform1 = RemoveSelfLoop()
        self.transform2 = ToSimple()

        if gnn_type == 'graph_conv':
            self.initial_emb1 = dglnn.GraphConv(in_dim, out_dim, allow_zero_in_degree=True)
            self.initial_emb2 = dglnn.GraphConv(out_dim, out_dim, allow_zero_in_degree=True)
            self.initial_emb3 = dglnn.GraphConv(out_dim, out_dim, allow_zero_in_degree=True)

            self.initial_emb21 = dglnn.GraphConv(out_dim, out_dim, allow_zero_in_degree=True)
            self.initial_emb22 = dglnn.GraphConv(out_dim, out_dim, allow_zero_in_degree=True)
            self.initial_emb23 = dglnn.GraphConv(out_dim, out_dim, allow_zero_in_degree=True)

            self.start_emb = dglnn.GraphConv(in_dim, out_dim, allow_zero_in_degree=True)
        elif gnn_type == 'gin_conv':
            self.initial_emb1 = GINConv(th.nn.Linear(in_dim, out_dim), 'max')
            self.initial_emb2 = GINConv(th.nn.Linear(out_dim, out_dim), 'max')
            self.initial_emb3 = GINConv(th.nn.Linear(out_dim, out_dim), 'max')

            self.initial_emb21 = GINConv(th.nn.Linear(out_dim, out_dim), 'max')
            self.initial_emb22 = GINConv(th.nn.Linear(out_dim, out_dim), 'max')
            self.initial_emb23 = GINConv(th.nn.Linear(out_dim, out_dim), 'max')

            self.initial_emb = GINConv(th.nn.Linear(out_dim, out_dim), 'max')

        elif gnn_type == 'gat_conv':
            self.initial_emb1 = GATConv(in_dim, out_dim, num_heads=3)
            self.initial_emb2 = GATConv(out_dim, out_dim, num_heads=3)
            self.initial_emb3 = GATConv(out_dim, out_dim, num_heads=3)

            self.initial_emb21 = GATConv(out_dim, out_dim, num_heads=3)
            self.initial_emb22 = GATConv(out_dim, out_dim, num_heads=3)
            self.initial_emb23 = GATConv(out_dim, out_dim, num_heads=3)

            self.initial_emb = GATConv(out_dim, out_dim, num_heads=3)
        else:
            self.initial_emb1 = TAGConv(in_dim, out_dim, k=2)
            self.initial_emb2 = TAGConv(out_dim, out_dim, k=2)
            self.initial_emb3 = TAGConv(out_dim, out_dim, k=2)

            self.initial_emb21 = TAGConv(out_dim, out_dim, k=2)
            self.initial_emb22 = TAGConv(out_dim, out_dim, k=2)
            self.initial_emb23 = TAGConv(out_dim, out_dim, k=2)

            self.initial_emb = TAGConv(out_dim, out_dim, k=2)

        self.dropout = nn.Dropout(0.2)

    def forward(self, graph, hx, eweight=None):
        scores = []
        node_sums = []
        h = self.dropout(F.relu(self.initial_emb1(graph, hx)))
        h = self.dropout(F.relu(self.initial_emb2(graph, h)))
        h = self.dropout(F.relu(self.initial_emb3(graph, h)))
        h = self.dropout(F.relu(self.initial_emb21(graph, h)))
        h = self.dropout(F.relu(self.initial_emb22(graph, h)))
        h = self.dropout(F.relu(self.initial_emb23(graph, h)))

        graph.ndata['new_feat'] = h
        # graphs = dgl.unbatch(graph)
        outs = dgl.mean_nodes(graph, 'new_feat')
        # print('comb')
        # print(torch.stack(outs))
        scores = self.pred_lin(outs)

        graph.ndata['h'] = graph.ndata['new_feat']
        if eweight is None:
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        else:
            graph.edata['w'] = eweight
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
        return scores


class edgepooling_training(nn.Module):

    def __init__(self, model, eps_in, n_class):
        super(edgepooling_training, self).__init__()

        # self.transform = RemoveSelfLoop()
        self.transform1 = RemoveSelfLoop()
        self.transform2 = ToSimple()

        self.dropout = nn.Dropout(0.2)
        self.n_class = n_class
        self.model = model
        self.eps_in = eps_in

    def get_score(self, bgraph, node_index_sum, score_map):
        key = str(node_index_sum.tolist())

        if key in score_map.keys():
            return score_map[key]
        else:
            x = [i for i in range(len(node_index_sum)) if node_index_sum[i] == 1]
            g = dgl.node_subgraph(bgraph, x)
            batch = dgl.batch([g])
            e_sigmoids = self.model(batch, batch.ndata['feat_onehot'])
            score_map[key] = e_sigmoids[0]
            return e_sigmoids[0]

    def forward_single_node(self, graph, h, node_id, kh):
        graph = dgl.khop_in_subgraph(graph, node_id, k=kh)[0]
        graph.ndata['node_index'] = torch.eye(len(graph.nodes())).to(device)
        total_ver = None
        first = True

        outs = []
        labels = []
        pooled_graph = graph

        nlclus = np.array(list(range(len(graph.nodes()))))

        nlclus_list = []
        pooled_graph_list = []
        pcluster_list = []

        last_nodes = pooled_graph.nodes()
        pooled_graph_features = h.float()

        pooled_graph = self.transform1(pooled_graph)

        pc = 0
        score_map = {}
        while True:
            if pooled_graph.num_edges() == 0:
                break;
            # print(pooled_graph)
            pooled_graph, pooled_graph_features, clusters, nlclus, label = self.pooling(graph, pooled_graph,
                                                                                        pooled_graph_features.float(),
                                                                                        nlclus, self.n_class, pc,
                                                                                        score_map)
            pc += 1

            pooled_graph = self.transform1(pooled_graph)

            if len(pooled_graph.nodes()) == len(last_nodes) and len(pcluster_list) > 0:
                break
            else:
                labels.append(label)
                nlclus_list.append(nlclus.copy())

                pooled_graph_list.append(pooled_graph)
                pcluster_list.append(clusters.copy())
                last_nodes = pooled_graph.nodes()

        return graph, outs, labels, nlclus_list, pcluster_list, pooled_graph_list

    def pooling(self, bgraph, graph, nodefeat, prev_node_lvl_cluster, num_class, pool_it, score_map):

        edge_cross_class = list(range(len(graph.edges()[0]))) * num_class
        edges = graph.edges()

        top_scores = []
        for x in range(num_class):
            e_sigmoids = []
            node_index_sum = graph.ndata['node_index']
            for ni in node_index_sum:
                e_sigmoids.append(self.get_score(bgraph, ni, score_map))

            e_sigmoids = torch.stack(e_sigmoids).squeeze(-1)

            enodes = []
            evotes = torch.softmax(e_sigmoids, dim=-1).max(dim=-1)[1]
            # print(graph)
            for i in torch.softmax(e_sigmoids, dim=-1):
                enodes.append(i[x])
            enodes = torch.stack(enodes)

            # print(enodes)
            esrc = enodes[edges[0]]
            edest = enodes[edges[1]]

            node_index_sum = graph.ndata['node_index'][graph.edges()[0]] + graph.ndata['node_index'][graph.edges()[1]]
            sub_gs = []

            ecomb_sigmoid = []

            node_count = []
            for ni in node_index_sum:
                ecomb_sigmoid.append(self.get_score(bgraph, ni, score_map))
                node_count.append(torch.sum(ni))
            node_count = torch.tensor(node_count).to(device)

            ecomb_sigmoid = torch.stack(ecomb_sigmoid).squeeze(-1)

            ecomb = []
            for i in torch.softmax(ecomb_sigmoid, dim=-1):
                ecomb.append(i[x])

            ecomb = torch.stack(ecomb)
            ecomb_sig = ecomb  # torch.sigmoid(ecomb)

            eps = self.eps_in/(pool_it + 1)#len(node_count)#0.001 / node_count
            multiplier = (self.eps_in)/(pool_it + 1)#len(node_count)
            hstc = esrc * torch.log(1 / (esrc)) + multiplier  # + (0.1/(pool_it+1))
            hdest = edest * torch.log(1 / (edest)) + multiplier  # + (0.1/(pool_it+1))
            hcomb = ecomb_sig * torch.log(1 / ecomb_sig) - eps  # - (1/(pool_it+1))


            scores = ((hstc - hcomb) * (hdest - hcomb) * (1 + torch.floor((hstc - hcomb))) * (
                        1 + torch.floor((hdest - hcomb))))
            top_scores = top_scores + scores.tolist()

        top_scores = torch.tensor(top_scores)
        perm = torch.argsort(top_scores.squeeze(), descending=True).tolist()

        new_src = []
        new_dst = []
        mask = torch.zeros(len(graph.nodes()), dtype=torch.bool).to(device)
        new_nodes = []
        new_label = []
        new_graphid = []
        new_edge_weights = []
        new_one_hot = []
        initial_c = max(prev_node_lvl_cluster)
        c = initial_c + 1
        cluster = [-1] * len(graph.nodes())
        # x.sort()
        selected_score = []
        # print('selected')
        for ex in perm:
            edge_index = edge_cross_class[ex]
            if scores[edge_index] <= 0:
                break

            node1 = graph.edges()[0][edge_index]
            node2 = graph.edges()[1][edge_index]

            if mask[node1] == 0 and mask[node2] == 0:
                selected_score.append(ecomb_sigmoid[edge_index])
                node_feat_1 = nodefeat[node1]
                node_feat_2 = nodefeat[node2]
                mask[node1] = 1
                mask[node2] = 1

                new_nodes.append((node_feat_1 + node_feat_2))  # * e[edge_index])

                new_one_hot.append(graph.ndata['node_index'][node1] + graph.ndata['node_index'][node2])

                cluster[node1] = c
                cluster[node2] = c
                prev_node_lvl_cluster[prev_node_lvl_cluster == node1.item()] = c
                prev_node_lvl_cluster[prev_node_lvl_cluster == node2.item()] = c
                c += 1
            mask[node1] = 1
            mask[node2] = 1
        c = max(c, 0)
        for i in range(0, len(cluster)):
            if cluster[i] == -1:
                cluster[i] = c
                prev_node_lvl_cluster[prev_node_lvl_cluster == i] = c
                new_one_hot.append(graph.ndata['node_index'][i])

                # new_graphid.append(graph.ndata['graphid'][i])
                # print(c)
                # print(prev_node_lvl_cluster)
                new_nodes.append(nodefeat[i])
                c += 1

        cluster = cluster - initial_c - 1
        prev_node_lvl_cluster = prev_node_lvl_cluster - initial_c - 1

        for ei in range(len(edges[0])):
            new_src.append(cluster[graph.edges()[0][ei]])
            new_dst.append(cluster[graph.edges()[1][ei]])

        new_g = dgl.graph((list(new_src), list(new_dst)), num_nodes=len(new_nodes))
        new_g = new_g.to(device)
        new_g.ndata["feat"] = torch.stack(new_nodes).double()
        new_g.edata['edge_score'] = ecomb_sigmoid
        new_g.ndata['node_index'] = torch.stack(new_one_hot)

        node_index_sum = new_g.ndata['node_index']  # + graph.ndata['node_index'][graph.edges()[1]]
        e_sigmoids = []
        for ni in node_index_sum:
            e_sigmoids.append(self.get_score(bgraph, ni, score_map))

        e_sigmoids = torch.stack(e_sigmoids).squeeze(-1)

        evotes = []
        for i in torch.softmax(e_sigmoids, dim=-1):
            evotes.append(i[i.max(dim=-1)[1].item()])

        evotes = torch.tensor(evotes).to(device)
        # evotes = evotes*torch.log(1/evotes)
        new_g.ndata['evotes'] = evotes


        return new_g, new_g.ndata["feat"], cluster, prev_node_lvl_cluster

    def forward(self, graph, h):

        graph.ndata['node_index'] = torch.eye(len(graph.nodes())).to(device)

        total_ver = None
        first = True

        outs = []
        labels = []
        pooled_graph = graph

        nlclus = np.array(list(range(len(graph.nodes()))))

        nlclus_list = []
        pooled_graph_list = []
        pcluster_list = []

        last_nodes = pooled_graph.nodes()
        pooled_graph_features = h.float()  # self.start_emb(pooled_graph, h.float())

        pooled_graph = self.transform1(pooled_graph)

        pc = 0
        score_map = {}
        while True:
            if pooled_graph.num_edges() == 0:
                break;
            # print(pooled_graph)
            pooled_graph, pooled_graph_features, clusters, nlclus = self.pooling(graph, pooled_graph,
                                                                                        pooled_graph_features.float(),
                                                                                        nlclus, self.n_class, pc,
                                                                                        score_map)
            pc += 1

            pooled_graph = self.transform1(pooled_graph)
            if len(pooled_graph.nodes()) == len(last_nodes) and len(pcluster_list) > 0:
                break
            else:
                nlclus_list.append(nlclus.copy())

                pooled_graph_list.append(pooled_graph)
                pcluster_list.append(clusters.copy())
                last_nodes = pooled_graph.nodes()
                # print(len(last_nodes))

        return outs, nlclus_list, pcluster_list, pooled_graph_list
