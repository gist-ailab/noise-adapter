import os
import torch
import argparse
import timm

import utils

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)

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
    wd = 5e-04
    lrde = [50, 75, 90]

    print(model_name, dataset_path.split('/')[-2], batch_size, class_range)
    
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError('save_path already exists')
    
    if 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar(args.data, dataset_path, batch_size)
        outlier_loader = utils.get_tinyimagenet('/SSDe/yyg', batch_size)
    elif 'svhn' == args.data:
        train_loader, valid_loader = utils.get_train_svhn(dataset_path, batch_size)
    elif 'domainnet' == args.data:        
        train_loader, valid_loader = utils.get_domainnet(dataset_path, 'A', 'real', batch_size)
    elif 'ham10000' == args.data:    
        train_loader, valid_loader = utils.get_imagenet('ham10000', dataset_path, batch_size)
    print(args.net)
    if 'resnet18' == args.net:
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)       
    if 'resnet18_order' == args.net:
        import resnet_order
        model = resnet_order.resnet18(num_classes=num_classes)  
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    # torch.save(model.state_dict(), save_path + '/start.pth.tar')
    state_dict = torch.load(config['save_path']+'/resnet18_baseline/last.pth.tar', map_location = device)['state_dict']
    model.load_state_dict(state_dict)
    model.eval()
    print(utils.validation_accuracy(model, valid_loader, device))
    
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay = wd)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lrde)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)   
    print(train_loader.dataset[0][0].shape)
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (data1, data2) in enumerate(zip(train_loader, outlier_loader)):
            inputs1, targets = data1
            inputs2, _  = data2
            inputs1, inputs2, targets = inputs1.to(device), inputs2.to(device), targets.to(device)
            optimizer.zero_grad()

            inputs = torch.cat([inputs1, inputs2], dim=0)
            
            x = model.forward_features(inputs)            
            features = model.global_pool(x).view(-1, 512)
            
            outputs = model.fc(features)
            
            loss = criterion(outputs[:len(targets)], targets)
            # loss += torch.norm(x[len(targets):], dim=[2,3], p=2).mean()
            loss += 0.5 * -(outputs[len(targets):].mean(1) - torch.logsumexp(outputs[len(targets):], dim=1)).mean()

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

        ## validation
        model.eval()
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