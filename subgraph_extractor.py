import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import AvgPooling, GNNExplainer
import copy

from models import pre_embedding, edgepooling_training
from imports import device
from dgl import RemoveSelfLoop
import dgl
from imports import *
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from dgl.data.utils import split_dataset
import argparse
from sklearn.model_selection import KFold

print(device)
#matplotlib.use("TkAgg")

def train(model, opt, curr_epoch, train_dataloader, val_loader, name, epochs):
    loss_func = F.nll_loss
    train_epoch_loss = []
    train_epoch_loss_layer = []
    test_epoch_loss = []
    val_epoch_loss = []

    best_acc = 0
    foldc = 0
    cluster_max_nodes = torch.tensor(8)
    lowest_test_loss = 999999999999
    for epoch in range(curr_epoch, epochs):
        running_loss_train = []
        running_loss_layer = []
        running_loss_test = []

        model.train()

        correct = 0
        for batchs, label in tqdm(train_dataloader):
            first = True

            score = model(batchs, batchs.ndata['feat_onehot'])
            var = torch.ones(batchs.batch_size, 2, requires_grad=True)
            loss = loss_func(F.log_softmax(score, dim=-1), label.squeeze(-1).type(torch.LongTensor).to(device))
            # loss = loss_func(F.softmax(score, dim=-1), F.one_hot(label.squeeze().long()).to(device), var)

            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += torch.sum(torch.eq(F.softmax(score, dim=-1).max(dim=-1)[1], label.squeeze().to(device)),
                                 -1).item()
            running_loss_train.append(loss.item())
        train_acc = correct / len(train_dataloader.dataset)

        correct = 0
        with torch.no_grad():
            for batchs, label in tqdm(val_loader):
                first = True

                score = model(batchs, batchs.ndata['feat_onehot'])
                var = torch.ones(batchs.batch_size, 2, requires_grad=True)
                val_loss = loss_func(F.log_softmax(score, dim=-1), label.squeeze(-1).type(torch.LongTensor).to(device))
                # loss = loss_func(F.softmax(score, dim=-1), F.one_hot(label.squeeze().long()).to(device), var)

                correct += torch.sum(torch.eq(F.softmax(score, dim=-1).max(dim=-1)[1], label.squeeze().to(device)),
                                     -1).item()
                running_loss_test.append(val_loss.item())
        val_acc = correct / len(val_loader.dataset)
        print('\ntrain accuracy = ' + str(train_acc) + '  val accuracy ' + str(val_acc))


        epoch_loss_train = np.mean(running_loss_train)
        # epoch_loss_layer = np.mean(running_loss_layer)
        epoch_loss_test = np.mean(running_loss_test)
        # epoch_loss_val = np.mean(running_loss_val)
        train_epoch_loss.append(epoch_loss_train)
        # train_epoch_loss_layer.append(epoch_loss_layer)
        test_epoch_loss.append(epoch_loss_test)

        if val_loss.item() < lowest_test_loss:
            lowest_test_loss = val_loss.item()
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'curr_epoch': epoch
            }, name + str(foldc) + ".pt")

        if epoch % 1 == 0:
            x = np.arange(0, len(train_epoch_loss))
            plt.title("Loss graph")
            plt.xlabel("X axis")
            plt.ylabel("Y axis")
            plt.plot(x, train_epoch_loss, color="red")
            plt.plot(x, test_epoch_loss, color="blue")
            plt.show()
            print(epoch_loss_train)

        print('---------------fold-' + str(foldc) + '-------- ecpoch ' + str(epoch) + " ------------------")
    return model

def pool(model, data, n):
    model.eval()
    pooler = edgepooling_training(model, n)
    for g,l in tqdm(data):
        feats = g.ndata['feat_onehot'].to(device)
        outs, nlclus_list, pcluster_list, pooled_graph_list = pooler(g, feats.detach().float())



def pool_only(path,data, hin, hout, n):
    model = pre_embedding(hin,hout, n).float().to(device) #todo
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    pooler = edgepooling_training(model, n)

    for g,l in tqdm(data):
        feats = g.ndata['feat_onehot'].to(device)
        outs, nlclus_list, pcluster_list, pooled_graph_list = pooler(g, feats.detach().float())

def start(dataset='MUTAG', dataset_feat='attr', dataset_multiplier=3, dw_dim=32, dw_walk_length= 10, dw_window_size=4, model_name='model_1', pool=True, epoch=800, hout=128, kfold=False):
    feat_key = dataset_feat
    if dataset == 'MUTAG':
        data = GINDataset('MUTAG', self_loop=True)
    else:
        data = GINDataset(dataset, self_loop=True)

    data = utils.prep_data(data, feat_key, dw_dim, dw_walk_length, dw_window_size) ##Deepwalk
    data = np.array(data)
    print("Data prepared")

    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    if kfold:
      k = 10
      folds = list(kf.split(data))[:]
    else:
      k = 1
      folds = list(kf.split(data))[:1]



    for train_ids, val_ids in folds:

        train_set = data[train_ids]
        val_set = data[val_ids]

        labels = set()

        for g,l in data:
            labels.add(l.item())
        labels = list(labels)
        labels.sort()
        print(labels)

        train_data = utils.get_supervised_dataloader(train_set,dataset_multiplier, feat_key, labels)
        val_data = utils.get_supervised_dataloader(val_set,dataset_multiplier, feat_key, labels)

        train_dataloader = GraphDataLoader(train_data, batch_size=64, shuffle=True)
        val_dataloader = GraphDataLoader(val_data, batch_size=64, shuffle=True)

        model = pre_embedding(39,hout, len(labels)).float().to(device) #todo
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        model = train(model, opt, 0, train_dataloader, val_dataloader, model_name, epoch)

        if pool:
            pool(model, train_data, len(labels))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", '--dataset',type=str, default='MUTAG')
    parser.add_argument("-dataset_feat", '--dataset_feat', type=str, default='attr')
    parser.add_argument("-dataset_multiplier", '--dataset_multiplier', type=int, default=3)
    parser.add_argument("-dw_dim", '--dw_dim', type=int, default=32)
    parser.add_argument("-dw_walk_length", '--dw_walk_length', type=int, default=10)
    parser.add_argument("-dw_window_size", '--dw_window_size', type=int, default=4)

    parser.add_argument("-model_name", '--model_name', type=str, default='model_1')
    parser.add_argument("-pool", '--pool', action='store_false')

    parser.add_argument("-epoch", '--epoch', type=int, default=800)
    parser.add_argument("-hin", '--hin', type=int, default=39)  #unused
    parser.add_argument("-hout", '--hout', type=int, default=128)
    parser.add_argument("-kfold", "--kfold", action='store_false')

    args = parser.parse_args()

    start(args.dataset, args.dataset_feat, args.dataset_multiplier, args.dw_dim, args.dw_walk_length, args.dw_window_size, args.model_name, args.pool, args.epoch, args.hout, args.kfold)














