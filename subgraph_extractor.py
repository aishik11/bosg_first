import dgl.function as fn
import torch
import torch.nn as nn
from dgl.data import GINDataset
from dgl.dataloading import GraphDataLoader
from dgl.nn import AvgPooling, GNNExplainer
import copy

from twisted.python.util import println

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
print(device)
matplotlib.use("TkAgg")

def train(model, opt, curr_epoch, train_dataloader, val_loader, name, epochs):
    loss_func = F.nll_loss
    train_epoch_loss = []
    train_epoch_loss_layer = []
    test_epoch_loss = []
    val_epoch_loss = []

    best_acc = 0
    foldc = 0
    cluster_max_nodes = torch.tensor(8)
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
                loss = loss_func(F.log_softmax(score, dim=-1), label.squeeze(-1).type(torch.LongTensor).to(device))
                # loss = loss_func(F.softmax(score, dim=-1), F.one_hot(label.squeeze().long()).to(device), var)

                correct += torch.sum(torch.eq(F.softmax(score, dim=-1).max(dim=-1)[1], label.squeeze().to(device)),
                                     -1).item()
                #running_loss_train.append(loss.item())
        val_acc = correct / len(val_loader.dataset)
        print('\ntrain accuracy = ' + str(train_acc) + '  val accuracy ' + str(val_acc))


        epoch_loss_train = np.mean(running_loss_train)
        # epoch_loss_layer = np.mean(running_loss_layer)
        epoch_loss_test = np.mean(running_loss_test)
        # epoch_loss_val = np.mean(running_loss_val)
        train_epoch_loss.append(epoch_loss_train)
        # train_epoch_loss_layer.append(epoch_loss_layer)
        test_epoch_loss.append(epoch_loss_test)
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
            plt.draw()
            print(epoch_loss_train)

        print('---------------fold-' + str(foldc) + '-------- ecpoch ' + str(epoch) + " ------------------")
    return model


if __name__ == '__main__':
    data = GINDataset('MUTAG', self_loop=True)
    data = utils.prep_data(data, 'attr')
    train_set, val_set = split_dataset(data,[0.8,0.2], True, 1)

    labels = set()

    for g,l in data:
        labels.add(l.item())
    labels = list(labels)
    labels.sort()
    print(labels)

    train_data = utils.get_supervised_dataloader(train_set,3, 'attr', labels)
    val_data = utils.get_supervised_dataloader(val_set,3, 'attr', labels)

    train_dataloader = GraphDataLoader(train_data, batch_size=64, shuffle=True)
    val_dataloader = GraphDataLoader(val_data, batch_size=64, shuffle=True)


    foldc = 0
    model = pre_embedding(39,128, 2).float().to(device)
    opt = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.001)
    model = train(model, opt, 0, train_dataloader, val_dataloader, 'model_1', 800)

    pooler = edgepooling_training(model, len(labels))

    for g,l in tqdm(train_data):
        feats = g.ndata['feat_onehot'].to(device)
        outs, nlclus_list, pcluster_list, pooled_graph_list = pooler(g, feats.detach().float())












