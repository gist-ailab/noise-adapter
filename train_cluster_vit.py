import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import timm
import numpy as np
import utils

from kmeans_pytorch import kmeans

def forward_embeddings(model, loader, device):
    embeddings = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            features_t = model.forward_features(inputs)
            features_t = model.global_pool(features_t).view(-1, 512)
            features_t = features_t / torch.norm(features_t, dim=1, keepdim=True)

            embeddings.append(features_t)
    embeddings = torch.cat(embeddings, dim=0)
    return embeddings


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'vit_tiny_patch16_224', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--nr', default = 0.2, type=float)

    args = parser.parse_args()
    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu

    model_name = args.net
    dataset_path = config['id_dataset']
    save_path = config['save_path'] + args.save_path
    num_classes = int(config['num_classes'])
    class_range = list(range(0, num_classes))

    batch_size = int(config['batch_size'])
    max_epoch = int(config['epoch'])
    lrde = [50, 75]
    num_clusters = num_classes

    print(model_name, dataset_path.split('/')[-2], batch_size, class_range)
    
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError('save_path already exists')
    
    if 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar_noisy(args.data, dataset_path, batch_size, args.nr)
    print(args.net)

    teacher = timm.create_model(args.net, pretrained=True, num_classes=num_classes)  
    teacher.to(device)
    teacher.eval()

    embeddings = forward_embeddings(teacher, train_loader, device)
    cluster_ids, cluster_centers = kmeans(X=embeddings, num_clusters = num_clusters, distance='euclidean', device= device)
    print(cluster_centers.shape)
    print(cluster_ids)

    train_loader = utils.modify_train_loader_with_cluster(train_loader, cluster_ids, cluster_centers)

    model = timm.create_model(args.net, pretrained=True, num_classes=num_classes)  
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    cos = torch.nn.CosineSimilarity()
    model.eval()
    print(utils.validation_accuracy(model, valid_loader, device))
    
    if 'vit' in args.net:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum=0.9, weight_decay = 1e-04)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay = 1e-04)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lrde)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)   
    print(train_loader.dataset[0][0].shape)
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets, cluster_ids, cluster_centers) in enumerate(train_loader):
            inputs, targets, cluster_ids, cluster_centers = inputs.to(device), targets.to(device), cluster_ids.to(device), cluster_centers.to(device)
            optimizer.zero_grad()

            features_s = model.forward_features(inputs)     
            features_s = model.global_pool(features_s).view(-1, 512)

            outputs = model.fc(features_s)
            

            ce_loss = criterion(outputs, targets)
            ce_loss = (ce_loss * cos(features_s, cluster_centers)).mean()
            
            features_s = features_s / torch.norm(features_s, dim=1, keepdim=True)
            cluster_loss = 5.0*(1 - cos(features_s, cluster_centers).mean())

            loss = ce_loss + cluster_loss
            loss.backward()            
            optimizer.step()

            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs[:len(targets)].max(1)            
            correct += predicted.eq(targets).sum().item()            
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        model.eval()
        ## Update Cluster Centers
        embeddings = []
        ids = []
        for batch_idx, (inputs, targets, cluster_ids, cluster_centers) in enumerate(train_loader):
            with torch.no_grad():
                inputs, targets, cluster_ids, cluster_centers = inputs.to(device), targets.to(device), cluster_ids.to(device), cluster_centers.to(device)
                features_t = model.forward_features(inputs)
                features_t = model.global_pool(features_t).view(-1, 512)
                features_t = features_t / torch.norm(features_t, dim=1, keepdim=True)
            embeddings.append(features_t)
            ids.append(cluster_ids)
        embeddings = torch.cat(embeddings, dim=0)
        ids = torch.cat(ids, dim=0)

        for i in range(num_clusters):
            cluster_embeddings = embeddings[ids == i]
            cluster_center = cluster_embeddings.mean(0)
            train_loader.dataset.cluster_centers[i]= cluster_center

        ## validation
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy(model, valid_loader, device)
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())
if __name__ =='__main__':
    train()