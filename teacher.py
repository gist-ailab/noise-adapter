import os
import numpy as np
import torch
import torch.nn as nn
import argparse
import timm
import matplotlib.pyplot as plt

from PIL import Image

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
    
    # if not os.path.exists(config['save_path']):
    #     os.mkdir(config['save_path'])
    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    # else:
    #     raise ValueError('save_path already exists')
    
    if 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar_jigsaw(args.data, dataset_path, batch_size)
    elif 'svhn' == args.data:
        train_loader, valid_loader = utils.get_train_svhn(dataset_path, batch_size)
    elif 'domainnet' == args.data:        
        train_loader, valid_loader = utils.get_domainnet(dataset_path, 'A', 'real', batch_size)
    elif 'ham10000' == args.data:    
        train_loader, valid_loader = utils.get_imagenet('ham10000', dataset_path, batch_size)
    elif 'imagenet' == args.data:
        train_loader, valid_loader = utils.get_imagenet('imagenet', dataset_path, batch_size)
    print(args.net)
    if 'resnet50' == args.net:
        teacher = timm.create_model(args.net, pretrained=True, num_classes=num_classes)
    teacher.to(device)
    teacher.eval()
    
    mean= torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
    std= torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
    upsampler = nn.Upsample([224,224], mode='nearest')
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)

            features = teacher.forward_features(inputs)
            features = features.mean(dim=1, keepdim=True)
            features_ = features.view(-1, 49)
            features = upsampler(features)
            print(features.shape)
            index = features_.sort(dim=1, descending=True).indices[0]
            # print(index[0][:2])
            
            image = inputs[0].cpu().permute(1,2,0)
            image = ((image*std)+mean) * 255
            
            img = Image.fromarray(image.numpy().astype(np.uint8))
            img.save('img_{}.png'.format(batch_idx))
            
            x1, y1 = int(index[0] / 7), int(index[0]%7)
            x2, y2 = int(index[1] / 7), int(index[1]%7)
            
            
            t1 = image[x1*32:(x1+1)*32, y1*32:(y1+1)*32].clone()
            t2 = image[x2*32:(x2+1)*32, y2*32:(y2+1)*32].clone()
            t3 = (t1+t2)/2
            
            image[x1*32:(x1+1)*32, y1*32:(y1+1)*32] = t2
            image[x2*32:(x2+1)*32, y2*32:(y2+1)*32] = t1   
            
            # print(t3.shape, image[x1*32:(x1+1)*32, y1*32:y1*32+2].shape, t3[:, 0:2].mean(1).shape)
            image[x1*32:(x1+1)*32, y1*32] = t3[:, 0:2].mean(dim=1)
            image[x1*32:(x1+1)*32, (y1+1)*32] = t3[:, -2:].mean(dim=1)
            image[x1*32, y1*32:(y1+1)*32] = t3[0:2, :].mean(dim=0)
            image[(x1+1)*32, y1*32:(y1+1)*32] = t3[-2:, :].mean(dim=0)
            
            image[x2*32:(x2+1)*32, y2*32] = t3[:, 0:2].mean(dim=1)
            image[x2*32:(x2+1)*32, (y2+1)*32] = t3[:, -2:].mean(dim=1)
            image[x2*32, y2*32:(y2+1)*32] = t3[0:2, :].mean(dim=0)
            image[(x2+1)*32, y2*32:(y2+1)*32] = t3[-2:, :].mean(dim=0)
                        
            
            img = Image.fromarray(image.numpy().astype(np.uint8))
            img.save('img_det_{}.png'.format(batch_idx))
            
            plt.imshow(features[0].cpu().squeeze())
            plt.savefig('heat_{}.png'.format(batch_idx))
            plt.cla()
            
            if batch_idx==10: break
            
        #       def __getitem__(self, index):
        # x, y = self.dataset[index]
        
        # s = int(float(x.size(1)) / 3)
        
        
        # x_ = torch.zeros_like(x)
        # tiles_order = random.sample(range(9), 9)
        # for o, tile in enumerate(tiles_order):
        #     i = int(o/3)
        #     j = int(o%3)
            
        #     ti = int(tile/3)
        #     tj = int(tile%3)
        #     # print(i, j, ti, tj)
        #     x_[:, i*s:(i+1)*s, j*s:(j+1)*s] = x[:, ti*s:(ti+1)*s, tj*s:(tj+1)*s] 
        # return x_, y  
            
if __name__ =='__main__':
    train()