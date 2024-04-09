import json
import torch.utils.data as data
import numpy as np
import torch
import os
import random

from PIL import Image
from scipy import io
from torchvision import transforms
from torchvision import datasets as dset
import torchvision

from .aptos import APTOS2019
from .chest14 import NIHchestXray
from .idrid import IDRID

def get_transform(transform_type='default', image_size=224, args=None):

    if transform_type == 'default':
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return train_transform, test_transform

def read_conf(json_path):
    """
    read json and return the configure as dictionary.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config
    
def get_dataset(path):
    train_transform, test_transform = get_transform()
    train_data = torchvision.datasets.ImageFolder(path + '/train', train_transform)
    valid_data = torchvision.datasets.ImageFolder(path + '/test', test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_noise_dataset(path, noise_rate = 0.2, batch_size = 32, seed = 0):
    train_transform, test_transform = get_transform()
    train_data = torchvision.datasets.ImageFolder(path + '/train', train_transform)
    np.random.seed(seed)
    
    new_data = []
    for i in range(len(train_data.samples)):
        if np.random.rand() > noise_rate: # clean sample:
            new_data.append([train_data.samples[i][0], train_data.samples[i][1]])
        else:
            label_index = list(range(7))
            label_index.remove(train_data.samples[i][1])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            new_data.append([train_data.samples[i][0], new_label])
    train_data.samples = new_data

    # Testing
    with open('label.txt', 'w') as f:
        for i in range(len(train_data.samples)):
            f.write('{}\n'.format(train_data.samples[i][1]))

    valid_data = torchvision.datasets.ImageFolder(path + '/test', test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_aptos_noise_dataset(path, noise_rate = 0.2, batch_size = 32, seed = 0):
    train_transform, test_transform = get_transform()
    train_data = APTOS2019(path, train=True, transforms = train_transform)

    np.random.seed(seed)
    new_data = []
    for i in range(len(train_data.samples)):
        if np.random.rand() > noise_rate: # clean sample:
            new_data.append([train_data.samples[i][0], train_data.samples[i][1]])
        else:
            label_index = list(range(5))
            label_index.remove(train_data.samples[i][1])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            new_data.append([train_data.samples[i][0], new_label])
    train_data.samples = new_data

    # Testing
    with open('label.txt', 'w') as f:
        for i in range(len(train_data.samples)):
            f.write('{}\n'.format(train_data.samples[i][1]))

    valid_data = APTOS2019(path, train=False, transforms = test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_idrid_noise_dataset(path, noise_rate = 0.2, batch_size = 32, seed = 0):
    train_transform, test_transform = get_transform()
    train_data = IDRID(path, train=True, transforms = train_transform)

    np.random.seed(seed)
    new_data = []
    for i in range(len(train_data.samples)):
        if np.random.rand() > noise_rate: # clean sample:
            new_data.append([train_data.samples[i][0], train_data.samples[i][1]])
        else:
            label_index = list(range(5))
            label_index.remove(train_data.samples[i][1])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            new_data.append([train_data.samples[i][0], new_label])
    train_data.samples = new_data

    # Testing
    with open('label.txt', 'w') as f:
        for i in range(len(train_data.samples)):
            f.write('{}\n'.format(train_data.samples[i][1]))

    valid_data = IDRID(path, train=False, transforms = test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader


def get_nihxray(path, batch_size = 32):
    train_transform, test_transform = get_transform()
    train_data = NIHchestXray(path, train=True, transforms = train_transform)
    valid_data = NIHchestXray(path, train=False, transforms = train_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader
    
if __name__ == '__main__':
    pass