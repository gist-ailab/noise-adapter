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

mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]

mean = [0, 0, 0]
std = [1, 1, 1]
size = 32

# , transforms.Normalize(mean, std)
train_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.RandomHorizontalFlip(), transforms.RandomCrop(size, padding=4),
                               transforms.ToTensor()])
# test_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])
devider = 1.5
test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])#, )

test_transform_gray = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1))])


resize_size = [224, 224]
crop_size = [224, 224]

class jigsaw_dataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y = self.dataset[index]
        
        s = int(float(x.size(1)) / 9)
        
        
        x_ = torch.zeros_like(x)
        tiles_order = random.sample(range(81), 81)
        for o, tile in enumerate(tiles_order):
            i = int(o/9)
            j = int(o%9)
            
            ti = int(tile/9)
            tj = int(tile%9)
            # print(i, j, ti, tj)
            x_[:, i*s:(i+1)*s, j*s:(j+1)*s] = x[:, ti*s:(ti+1)*s, tj*s:(tj+1)*s] 
        return x_, y
        
def get_cifar_test(dataset, folder, batch_size, test=False):
    test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor()])
    test_transform_cifar_blur = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor()])
    train_ = not test
    
    if dataset == 'cifar10':
        train_data = dset.CIFAR10(folder, train=train_, transform=test_transform_cifar, download=True)
        test_data = dset.CIFAR10(folder, train=train_, transform=test_transform_cifar_blur, download=True)

    else:
        train_data = dset.CIFAR100(folder, train=train_, transform=test_transform_cifar, download=True)
        test_data = dset.CIFAR100(folder, train=train_, transform=test_transform_cifar_blur, download=True)
    jigsaw = jigsaw_dataset(test_data)
    # print(jigsaw[0])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(jigsaw, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, valid_loader


test_transform_224_gray = transforms.Compose([transforms.Resize(resize_size),
                        transforms.CenterCrop(crop_size), 
                        transforms.ToTensor(), 
                        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std= [0.229, 0.224, 0.225])]
                        )

train_transforms = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),

    # transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225]) 
])

# test_transforms = transforms.Compose([
#     transforms.Resize(resize_size),
#     transforms.CenterCrop(crop_size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                         std= [0.229, 0.224, 0.225]),
#     # transforms.Grayscale(3)
# ])
def get_few_shot_svhn_loader(path, shot=1):
    test_data = dset.SVHN(path, split='test', transform=test_transform_cifar, download=False)
    # print(test_data.data.shape, random.sample(list(range(len(test_data))), k=shot))
    test_data.data = test_data.data[random.sample(list(range(len(test_data))), k=shot)]
    # print(len(test_data))
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False, pin_memory=True, num_workers = 4)    
    return valid_loader
    
def get_few_shot_loader(path, shot=1):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_data.samples = random.sample(ood_data.samples, shot)

    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader



test_transforms = test_transform_cifar


def get_imagenet(dataset, imagenet_path, batch_size=32, eval=False):
    train_trans = train_transforms
    test_trans = test_transforms
    if eval:
        train_trans = test_transforms

    trainset = torchvision.datasets.ImageFolder(imagenet_path+'/train', train_trans)
    testset = torchvision.datasets.ImageFolder(imagenet_path+'/val', test_trans)
    # trainset = jigsaw_dataset(trainset)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def read_conf(json_path):
    """
    read json and return the configure as dictionary.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config
    
def get_cifar(dataset, folder, batch_size, eval=False):
    if eval==True:
        train_transform_cifar_ = test_transform_cifar
    else:
        train_transform_cifar_ = train_transform_cifar
    if dataset == 'cifar10':
        train_data = dset.CIFAR10(folder, train=True, transform=train_transform_cifar_, download=True)
        test_data = dset.CIFAR10(folder, train=False, transform=test_transform_cifar, download=True)
        num_classes = 10
    else:
        train_data = dset.CIFAR100(folder, train=True, transform=train_transform_cifar_, download=True)
        test_data = dset.CIFAR100(folder, train=False, transform=test_transform_cifar, download=True)
        num_classes = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, valid_loader

def get_train_svhn(folder, batch_size):
    train_data = dset.SVHN(folder, split='train', transform=test_transform_cifar, download=True)    
    test_data = dset.SVHN(folder, split='test', transform=test_transform_cifar, download=True)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)     
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return train_loader, valid_loader
    
def get_outlier(path, batch_size):
    class temp(torch.utils.data.Dataset):
        def __init__(self, path, transform=None):
            self.data = np.load(path)
            self.transform = transform

        def __getitem__(self, index):
            data = self.data[index]
            data = self.transform(data)
            return data

        def __len__(self):
            return len(self.data)
    
    test_data = temp(path, transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
     transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4)    
    return valid_loader

def get_tinyimagenet(path, batch_size):
    class TinyImages(torch.utils.data.Dataset):
        def __init__(self, path, transform=None, exclude_cifar=True):
            data_file = open(path+'/', "rb")

            def load_image(idx):
                data_file.seek(idx * 3072)
                data = data_file.read(3072)
                return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

            self.load_image = load_image
            self.offset = 0     # offset index

            self.transform = transform
            self.exclude_cifar = exclude_cifar

            if exclude_cifar:
                self.cifar_idxs = []
                with open(path+'/80mn_cifar_idxs.txt', 'r') as idxs:
                    for idx in idxs:
                        # indices in file take the 80mn database to start at 1, hence "- 1"
                        self.cifar_idxs.append(int(idx) - 1)

                # hash table option
                self.cifar_idxs = set(self.cifar_idxs)
                self.in_cifar = lambda x: x in self.cifar_idxs

        def __getitem__(self, index):
            index = (index + self.offset) % 79302016

            if self.exclude_cifar:
                while self.in_cifar(index):
                    index = np.random.randint(79302017)

            img = self.load_image(index)
            if self.transform is not None:
                img = self.transform(img)

            return img, 0  # 0 is the class
        def __len__(self):
            return 79302017

    ood_data = TinyImages(path, test_transform_cifar, True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    return ood_loader

def get_svhn(folder, batch_size, transform_imagenet = False):
    test_data = dset.SVHN(folder, split='test', transform=test_transforms if transform_imagenet else test_transform_cifar, download=True)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return valid_loader

def get_svhn_test(folder, batch_size):
    test_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])
    test_transform_cifar_blur = transforms.Compose([transforms.Resize([4,4]), transforms.Resize([32,32]), transforms.ToTensor()])

    test_data = dset.SVHN(folder, split='test', transform=test_transform_cifar, download=True)
    test_data_blur = dset.SVHN(folder, split='test', transform=test_transform_cifar_blur, download=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    blur_loader = torch.utils.data.DataLoader(test_data_blur, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    

    
    return test_loader, blur_loader

def get_textures(path):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=128, shuffle=False, pin_memory=True)
    return ood_loader

def get_ood_blur(path):
    test_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])
    test_transform_cifar_blur = transforms.Compose([transforms.Resize([4,4]), transforms.Resize([32,32]), transforms.ToTensor()])

    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_data_blur = torchvision.datasets.ImageFolder(path, test_transform_cifar_blur)

    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    ood_loader_blur = torch.utils.data.DataLoader(ood_data_blur, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader, ood_loader_blur

def get_lsun(path):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader    

def get_places_blur(path):
    test_transform_cifar = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor()])
    test_transform_cifar_blur = transforms.Compose([transforms.Resize([4,4]), transforms.Resize([32,32]), transforms.ToTensor()])

    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_data_blur = torchvision.datasets.ImageFolder(path, test_transform_cifar_blur)

    random.seed(0)
    ood_data.samples = random.sample(ood_data.samples, 10000)
    ood_data_blur.samples = random.sample(ood_data.samples, 10000)

    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    ood_loader_blur = torch.utils.data.DataLoader(ood_data_blur, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader, ood_loader_blur

def get_places(path):
    test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])

    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)

    random.seed(0)
    ood_data.samples = random.sample(ood_data.samples, 10000)

    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader

def get_mnist(path, batch_size=100, transform_imagenet = False):
    ood_data = dset.MNIST(path, train=False, transform=test_transform_224_gray if transform_imagenet else test_transform_gray, download=True)
    print(ood_data[0][0].shape)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    return ood_loader 

def get_knist(path):
    ood_data = dset.KMNIST(path, train=False, transform=test_transform_gray, download=True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader 

def get_fnist(path):
    ood_data = dset.FashionMNIST(path, train=False, transform=test_transform_gray, download=True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader 

def get_food101(path):
    ood_data = dset.Food101(path, split='test', transform=test_transform_cifar, download=True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader 

def get_stl10(path):
    ood_data = dset.STL10(path, split='test', transform=test_transform_cifar, download=True)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader 


def get_folder(path):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader 

def get_blob():
    # /////////////// Blob ///////////////
    ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(10000, 32, 32, 3)))
    for i in range(10000):
        ood_data[i] = gblur(ood_data[i], sigma=1.5, channel_axis=False)
        ood_data[i][ood_data[i] < 0.75] = 0.0

    dummy_targets = torch.ones(10000)
    ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=True, pin_memory=True)
    return ood_loader

def get_gaussian():
    dummy_targets = torch.ones(50000)
    gaussian = np.random.randint(0, 255, [50000, 3, 32, 32], dtype=int)
    gaussian = gaussian/255.0
    # print(gaussian)
    ood_data = torch.from_numpy(np.float32(gaussian))
    # ood_data = torch.from_numpy(np.float32(np.clip(
    #     np.ones([50000, 3, 32, 32]), 0, 1))
    # )
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size = 128, shuffle=True)
    return ood_loader

def get_rademacher():
    dummy_targets = torch.ones(10000)
    ood_data = torch.from_numpy(np.random.binomial(
        n=1, p=0.5, size=(10000, 3, 32, 32)).astype(np.float32)) * 2 - 1
    ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=True)
    return ood_loader


def get_domainnet(path, split, subset, batch_size, eval=False):
    train_trans = train_transforms
    test_trans = test_transforms
    if eval:
        train_trans = test_transforms
    trainset = DomainNetClass(path, split, subset, train=True, transform=train_trans)
    testset = DomainNetClass(path, split, subset, train=False, transform=test_trans)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, pin_memory=True, num_workers = 8)
    valid_loader = torch.utils.data.DataLoader(testset, batch_size, shuffle=False, pin_memory=True, num_workers = 8)
    return train_loader, valid_loader

def get_ood_folder(path, batch_size = 32, sample_cut = False):
    oodset = torchvision.datasets.ImageFolder(path, test_transforms)
    if sample_cut:
        oodset.samples = oodset.samples[:3000]
    ood_loader = torch.utils.data.DataLoader(oodset, batch_size, shuffle = True, pin_memory = True, num_workers = 4)
    return ood_loader
    
if __name__ == '__main__':
    pass