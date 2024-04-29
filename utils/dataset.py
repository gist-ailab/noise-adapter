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

from .aptos import APTOS2019, APTOS2019_valid, APTOS2019TwoLabel
from .chest14 import NIHchestXray
from .idrid import IDRID
from .chaoyang import CHAOYANG
from .dr import DR

class ImageFolderTwoLabel(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        clean_label = self.ori_labels[index]

        return sample, target, clean_label
    
def get_noise_dataset_with_cleanlabel(path, noise_rate = 0.2, batch_size = 32, seed = 0):
    train_transform, test_transform = get_transform()
    train_data = ImageFolderTwoLabel(path + '/train', train_transform)
    np.random.seed(seed)
    
    new_data = []
    ori_labels = []
    for i in range(len(train_data.samples)):
        ori_labels.append(train_data.samples[i][1])
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
    train_data.ori_labels = ori_labels

    # Testing
    with open('label.txt', 'w') as f:
        for i in range(len(train_data.samples)):
            f.write('{}\n'.format(train_data.samples[i][1]))

    valid_data = torchvision.datasets.ImageFolder(path + '/test', test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_aptos_noise_dataset_with_cleanlabel(path, noise_rate = 0.2, batch_size = 32, seed = 0):
    train_transform, test_transform = get_transform()
    train_data = APTOS2019TwoLabel(path, train=True, transforms = train_transform)

    np.random.seed(seed)
    new_data = []
    ori_labels = []
    for i in range(len(train_data.samples)):
        ori_labels.append(train_data.samples[i][1])
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
    train_data.ori_labels = ori_labels

    # Testing
    with open('label.txt', 'w') as f:
        for i in range(len(train_data.samples)):
            f.write('{}\n'.format(train_data.samples[i][1]))

    valid_data = APTOS2019(path, train=False, transforms = test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader


def get_transform(transform_type='default', image_size=224, args=None):

    if transform_type == 'default':
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter(),
            transforms.RandomCrop(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((image_size, image_size)),
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



def get_mnist_noise_dataset(dataname, noise_rate = 0.2, batch_size = 32, seed = 0):
    # from medmnist import NoduleMNIST3D
    from medmnist import PathMNIST, BloodMNIST, OCTMNIST, TissueMNIST, OrganCMNIST
    train_transform, test_transform = get_transform()

    if dataname == 'pathmnist':
        train_data = PathMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = PathMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 9
    if dataname == 'bloodmnist':
        train_data = BloodMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = BloodMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 8
    if dataname == 'octmnist':
        train_data = OCTMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = OCTMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 4
    if dataname == 'tissuemnist':
        train_data = TissueMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = TissueMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 8
    if dataname == 'organcmnist':
        train_data = OrganCMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
        test_data = OrganCMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)
        num_classes = 11

    np.random.seed(seed)
    # new_imgs = []
    new_labels =[]
    for i in range(len(train_data.imgs)):
        if np.random.rand() > noise_rate: # clean sample:
            # new_imgs.append(train_data.imgs[i])
            new_labels.append(train_data.labels[i][0])
        else:
            label_index = list(range(num_classes))
            label_index.remove(train_data.labels[i])
            label_index = np.array(label_index)
            label_index = np.reshape(label_index, (-1))

            new_label = np.random.choice(label_index, 1)
            new_label = new_label[0]
            
            # new_imgs.append(train_data.imgs[i])
            new_labels.append(new_label)
    # train_data.imgs = new_imgs
    train_data.labels = new_labels

    new_labels = []
    for i in range(len(test_data.labels)):
        new_labels.append(test_data.labels[i][0])
    test_data.labels = new_labels

    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
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

def get_chaoyang_dataset(path, batch_size = 32, seed = 0):
    train_transform, test_transform = get_transform()
    train_data = CHAOYANG(path, train=True, transforms = train_transform)
    valid_data = CHAOYANG(path, train=False, transforms = test_transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_dr(path, batch_size = 32):
    train_transform, test_transform = get_transform()

    train_data = DR(path, train=True, transforms = train_transform)
    aptos_train_data = APTOS2019('./data/APTOS-2019', train=True, transforms = test_transform)
    aptos_valid_data = APTOS2019_valid('./data/APTOS-2019', transforms = test_transform)
    aptos_test_data = APTOS2019('./data/APTOS-2019', train=False, transforms = test_transform)



    aptos_train_data.samples.extend(aptos_valid_data.samples)
    aptos_train_data.samples.extend(aptos_test_data.samples)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(aptos_train_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_nihxray(batch_size = 32):
    from medmnist import ChestMNIST

    train_transform, test_transform = get_transform()

    def __modify_len__(self):
        return self.imgs.shape[0]
    ChestMNIST.__len__ = __modify_len__
    train_data = ChestMNIST(split="train", download=True, size=224, transform= train_transform, as_rgb=True)
    test_data = ChestMNIST(split="test", download=True, size=224, transform= test_transform, as_rgb=True)

    # print(train_data.imgs)
    # print(train_data.labels)
    multi2single = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ,13 ,14]

    imgs = []
    labels = []
    for i in range(len(train_data.imgs)):
        if (train_data.labels[i] != 0).sum() > 1:
            pass
        else:
            imgs.append([train_data.imgs[i]])
            label = ((train_data.labels[i] != 0) * multi2single).sum()
            labels.append([label])
    train_data.imgs = np.concatenate(imgs)
    train_data.labels = np.concatenate(labels)
    
    # print(train_data.imgs.shape)
    # print(train_data.labels.shape)


    imgs = []
    labels = []
    for i in range(len(test_data.imgs)):
        if (test_data.labels[i] != 0).sum() > 1:
            pass
        else:
            imgs.append([test_data.imgs[i]])
            label = ((test_data.labels[i] != 0) * multi2single).sum()
            labels.append([label])
    test_data.imgs = np.concatenate(imgs)
    test_data.labels = np.concatenate(labels)

    # print(test_data.imgs.shape)
    # print(test_data.labels.shape)
    # print(test_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers = 16)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader
    
if __name__ == '__main__':
    get_nihxray()