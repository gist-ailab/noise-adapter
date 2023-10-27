import os
import torch
import argparse
import timm
import numpy as np
import utils

import random

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns 

import inspect

def forward_features(model, inputs):
    out = model.forward_features(inputs)
    out = model.global_pool(out).view(-1, 512) # Class token
    out = model(inputs)
    # For vit
    x = model.patch_embed(inputs)
    x = model._pos_embed(x)
    x = model.norm_pre(x)
    x = model.blocks[:-1](x)
    x = model.blocks[-1].norm1(x)
    B, N, C = x.shape
    # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    # print( model.blocks[-1].attn.num_heads, model.blocks[-1].attn.num_heads)
    qkv = model.blocks[-1].attn.qkv(x).reshape(B, N, 3, model.blocks[-1].attn.num_heads, 64).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    # q, k = model.blocks[-1].attn.q_norm(q), model.blocks[-1].attn.k_norm(k)
    
    # print(qkv.shape, q.shape, k.shape, v.shape)
    k = k.mean(1).mean(1)
    print(k.shape)
    return out

def calculate_embeddings(model, loader, device):
    id_features = []
    targets = []

    for batch_idx, (inputs, target) in enumerate(loader):
        with torch.no_grad():
            inputs, target = inputs.to(device), target.to(device)
            targets.append(target)

            feature = forward_features(model, inputs).cpu()
            feature = feature / torch.norm(feature, dim=1, keepdim=True)
            id_features.append(feature)
    id_features = torch.cat(id_features, dim=0)
    targets = torch.cat(targets, dim=0)
    return id_features, targets

def tsne(id_features, targets):
    tsne = TSNE(n_components=2, perplexity=50, early_exaggeration = 24.0)
    num_classes = 10
    features = id_features.cpu()[:10000]
    targets = targets.cpu()[:10000]
    targets2 = np.random.permutation(targets)

    targets2 = torch.cat([targets[:5000], torch.tensor(targets2)[:5000]], dim=0)
    print(features.shape, targets.shape)

    tsne_results = tsne.fit_transform(features.cpu().numpy())

    x, y = np.split(tsne_results, 2, axis=1)
    print(x.shape, y.shape)

    plt.figure(dpi=1000)
    colors = ['darkred', 'chocolate', 'darkgoldenrod', 'olivedrab', 'forestgreen', 'lightseagreen', 'darkslategray', 'dodgerblue', 'mediumpurple', 'mediumvioletred']
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']     
    for i in range(num_classes):
        x_ = x[targets==i]
        y_ = y[targets==i]
        plt.scatter(x_[250:750], y_[250:750], c = colors[i], alpha=0.8, s= 5, marker = 'o', edgecolors='k', linewidths=0.5, label = labels[i])
    plt.legend()
    plt.savefig('tsne_.png')
    plt.cla()

    plt.figure(dpi=1000)
    colors = ['darkred', 'chocolate', 'darkgoldenrod', 'olivedrab', 'forestgreen', 'lightseagreen', 'darkslategray', 'dodgerblue', 'mediumpurple', 'mediumvioletred']
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']     
    for i in range(num_classes):
        x_ = x[targets2==i]
        y_ = y[targets2==i]
        plt.scatter(x_[250:750], y_[250:750], c = colors[i], alpha=0.8, s= 5, marker = 'o', edgecolors='k', linewidths=0.5, label = labels[i])
    plt.legend()
    plt.savefig('tsne_noise.png')


def eval():
    sns.set_style("darkgrid")
    # sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'vit_tiny_patch16_224', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', type=str)
    parser.add_argument('--save_path', '-s', type=str)

    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    dataset_path = config['id_dataset']
    batch_size = config['batch_size']
    # save_path = config['save_path'] + args.save_path
    
    num_classes = int(config['num_classes'])

    train_loader, valid_loader = utils.get_cifar_noisy(args.data, dataset_path, batch_size, 0.0)
    model =timm.create_model(args.net, pretrained=True, num_classes=num_classes)
    model.to(device)
    model.eval()

    valid_accuracy = utils.validation_accuracy(model, valid_loader, device)
    print('In-distribution accuracy: ', valid_accuracy)
        
    id_features, targets = calculate_embeddings(model, train_loader, device)
    print(id_features.shape, targets.shape)

    tsne(id_features, targets)

if __name__ =='__main__':
    eval()