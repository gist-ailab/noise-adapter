import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import timm
import numpy as np
import utils
import models

# def random_masking(x, masked_ratio=0.2):
#     """
#     Produces masked image
#     """
#     B, C, W, H = x.shape  # batch, length, dim

#     x_masked = x.clone()
#     jigsaw_num = 8 #np.random.choice([3])

#     s = int(W / jigsaw_num)

#     # masked tile selection
#     masked_tile = torch.argsort(torch.rand([B, jigsaw_num * jigsaw_num]), dim=1)
#     masked_tile = masked_tile[:, :int(masked_tile.size(1)*masked_ratio)]

#     for b in range(B):
#         for tile in masked_tile[b]:
#             i = int(tile/jigsaw_num)
#             j = int(tile%jigsaw_num)
#             x_masked[b, :, i*s:(i+1)*s, j*s:(j+1)*s] = 0
#     return x_masked
def random_masking(x, masked_ratio=0.01):
    """
    Produces masked image
    """
    B, C, W, H = x.shape  # batch, length, dim

    x_masked = x.clone()
    x_masked = x_masked + masked_ratio*(0.1**0.5)*torch.randn([B,C,W,H]).to(x.device)
    return x_masked


def softmax_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

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

    if args.net == 'resnet18':
        model = models.ResNet18(num_classes=1000)
        model.load_state_dict(torch.load('/SSDe/yyg/RR/pretrained_resnet18/last.pth.tar', map_location=device)['state_dict'])
        model.fc = torch.nn.Linear(512, num_classes)
    model.to(device)

    ema_model = timm.utils.ModelEmaV2(model, decay = 0.999, device = device)
    

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    print(utils.validation_accuracy(model, valid_loader, device))

    if 'vit' in args.net:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay = 1e-04)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9, weight_decay = 1e-04)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lrde)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)   
    print(train_loader.dataset[0][0].shape)

    f = open(save_path + '/record.txt', 'w')
    ce_lambda = 1.0
    check = False
    for epoch in range(max_epoch):
        ## training
        model.train()
        ema_model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs_ema = ema_model.module(inputs)    

            if check:
                with torch.no_grad():
                    outputs_ema_avg = []
                    for i in range(10):
                        ema_model.eval()
                        masked_inputs = random_masking(inputs, 0.05)

                        outputs_ema_ = ema_model.module(masked_inputs)
                        outputs_ema_avg.append(outputs_ema_.unsqueeze(0))

                        # from PIL import Image
                        # masked_inputs = (masked_inputs[0].permute(1, 2, 0) * 0.5 + 0.5) *  255.0
                        # img = Image.fromarray(np.array(masked_inputs.cpu().numpy(), dtype=np.uint8))
                        # img.save('test.png')

                        # masked_inputs = (inputs[0].permute(1, 2, 0) * 0.5 + 0.5) *  255.0
                        # img = Image.fromarray(np.array(masked_inputs.cpu().numpy(), dtype=np.uint8))
                        # img.save('test_ori.png')

                outputs_ema_avg = torch.cat(outputs_ema_avg, dim=0)
                outputs_ema_avg = outputs_ema_avg.mean(dim=0)

                pseudo_label = outputs_ema.max(dim=1).indices
                ce_loss = criterion(outputs, pseudo_label)
                consistency_loss = softmax_entropy(outputs, outputs_ema_avg).mean()
            else:
                ce_loss = criterion(outputs, targets)
                consistency_loss = softmax_entropy(outputs, outputs_ema).mean()            
            loss = ce_loss + consistency_loss
            loss.backward()            
            optimizer.step()

            ema_model.update(model)
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
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy(model, valid_loader, device)
        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        ema_accuracy = utils.validation_accuracy(ema_model.module, train_loader, device)
        
        if ema_accuracy < (train_accuracy + 0.005) and ema_accuracy > (train_accuracy - 0.005) and not check:
            check = True

        print(ema_accuracy, train_accuracy, check)
        # print()
        valid_accuracy_ema = utils.validation_accuracy(ema_model.module, valid_loader, device)
        print(valid_accuracy_ema)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())
        f.write('{}\t{}\t{}\t{}\t{}\n'.format(epoch, ema_accuracy, train_accuracy, valid_accuracy_ema, valid_accuracy))
    f.close()

if __name__ =='__main__':
    train()