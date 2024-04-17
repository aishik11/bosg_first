import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
from dgl.data import GINDataset,TUDataset
from dgl.dataloading import GraphDataLoader
import copy
import torch.nn.functional as F

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
from dgl.data import BA2MotifDataset

print(device)
#matplotlib.use("TkAgg")

def train(model, opt, curr_epoch, train_dataloader, val_loader, path, k, gnntype, dataset_name, epochs, halt_pat):

    print([gnntype, dataset_name])
    loss_func = F.nll_loss
    train_epoch_loss = []
    train_epoch_loss_layer = []
    test_epoch_loss = []
    val_epoch_loss = []

    best_acc = 0
    foldc = 0
    cluster_max_nodes = torch.tensor(8)
    lowest_test_loss = 999999999999

    halt_counter = 0
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
        halt_counter +=1
        if val_loss.item() < lowest_test_loss:
            halt_counter = 0
            lowest_test_loss = val_loss.item()
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'curr_epoch': epoch
            }, os.path.join(path, gnntype +"_" + dataset_name +"_"+ str(k) + ".pt"))

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
        if halt_counter > halt_pat:
            break
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

def start(dataset_name='MUTAG', dataset = None, dataset_feat='attr', dataset_multiplier=3, dw_dim=32, dw_walk_length= 10, dw_window_size=4, model_dir='', pool=True, epoch=800, hout=128, kfold=False,halt_pat=50):
    feat_key = dataset_feat

    ##not used
    data = []
    if dataset_name == 'MUTAG':
        tdata = TUDataset('MUTAG')
        
        for g,l in tdata:
          ng = dgl.graph(((g.edges()[0]).int().tolist(), ( g.edges()[1]).int().tolist()))
          ng.ndata['attr'] = F.one_hot(g.ndata['node_labels'].squeeze(),num_classes=7)
          data.append((ng,l))
          #print(ng)
    elif dataset_name == 'BASHAPE':
        dataset = BA2MotifDataset()
        data = []
        for g, l in dataset:
          data.append((g, torch.argmax(l)))
        feat_key = 'feat'
    else:
        data = dataset#GINDataset(dataset, self_loop=True)


    print(len(data))
    data = utils.prep_data(data, feat_key, dw_dim, dw_walk_length, dw_window_size)
    print(data[0]) ##Deepwalk
    data = np.asarray(data, dtype="object")
    print("Data prepared")

    kf = KFold(n_splits=10, random_state=1, shuffle=True)
    if kfold:
      k = 10
      folds = list(kf.split(data))[:]
    else:
      k = 1
      folds = list(kf.split(data))[:1]


    labels = set()

    for g,l in data:
        labels.add(l.item())
    labels = list(labels)
    labels.sort()
    print(labels)

    for train_ids, val_ids in folds:

        train_set = data[train_ids]
        val_set = data[val_ids]



        train_data = utils.get_supervised_dataloader(train_set,dataset_multiplier, feat_key, labels)
        val_data = utils.get_supervised_dataloader(val_set,dataset_multiplier, feat_key, labels)

        train_dataloader = GraphDataLoader(train_data, batch_size=64, shuffle=True)
        val_dataloader = GraphDataLoader(val_data, batch_size=64, shuffle=True)


        indim = val_data[0][0].ndata['feat_onehot'].size()[1]

        for gnntype in ['graph_conv', 'gin_conv', 'gat_conv', 'tag_conv']:
            model = pre_embedding(indim,hout, len(labels), gnntype).float().to(device) #todo
            opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
            model = train(model, opt, 0, train_dataloader, val_dataloader, model_dir, k, gnntype, dataset_name, epoch, halt_pat)

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

    parser.add_argument("-model_dir", '--model_dir', type=str, default='./')
    parser.add_argument("-pool", '--pool', action='store_false')

    parser.add_argument("-epoch", '--epoch', type=int, default=800)
    parser.add_argument("-hin", '--hin', type=int, default=39)  #unused
    parser.add_argument("-hout", '--hout', type=int, default=128)
    parser.add_argument("-kfold", "--kfold", action='store_false')

    args = parser.parse_args()

    start(args.dataset, args.dataset_feat, args.dataset_multiplier, args.dw_dim, args.dw_walk_length, args.dw_window_size, args.model_dir, args.pool, args.epoch, args.hout, args.kfold)














