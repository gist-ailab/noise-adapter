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

# mean = [0, 0, 0]
# std = [1, 1, 1]
size = 32

# , transforms.Normalize(mean, std)
train_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.RandomHorizontalFlip(), transforms.RandomCrop(size, padding=4),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])
test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])#, )

test_transform_gray = transforms.Compose([transforms.Resize([32,32]), transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1))])


resize_size = [224, 224]
crop_size = [224, 224]

class jigsaw_train_dataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.blur_transforms = transforms.Compose([transforms.Resize([2, 2]), transforms.Resize([size,size]), transforms.RandomHorizontalFlip(), transforms.RandomCrop(size, padding=4),
                               transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y = self.dataset[index]
        img = self.dataset.data[index]
        img = Image.fromarray(img)
        img = self.blur_transforms(img)        
        s = int(float(x.size(1)) / 3)
        
        
        x_ = torch.zeros_like(x)
        tiles_order = random.sample(range(9), 9)
        for o, tile in enumerate(tiles_order):
            i = int(o/3)
            j = int(o%3)
            
            ti = int(tile/3)
            tj = int(tile%3)
            # print(i, j, ti, tj)
            x_[:, i*s:(i+1)*s, j*s:(j+1)*s] = x[:, ti*s:(ti+1)*s, tj*s:(tj+1)*s] 
        return x, x_, img, y
    
def get_cifar_jigsaw(dataset, folder, batch_size):
    
    if dataset == 'cifar10':
        train_data = dset.CIFAR10(folder, train=True, transform=train_transform_cifar, download=True)
        test_data = dset.CIFAR10(folder, train=False, transform=test_transform_cifar, download=True)

    else:
        train_data = dset.CIFAR100(folder, train=True, transform=train_transform_cifar, download=True)
        test_data = dset.CIFAR100(folder, train=False, transform=test_transform_cifar, download=True)
    jigsaw = jigsaw_train_dataset(train_data)
    # print(jigsaw[0])

    train_loader = torch.utils.data.DataLoader(jigsaw, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, valid_loader

class jigsaw_dataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y = self.dataset[index]
        
        s = int(float(x.size(1)) / 3)
        
        
        x_ = torch.zeros_like(x)
        tiles_order = random.sample(range(9), 9)
        for o, tile in enumerate(tiles_order):
            i = int(o/3)
            j = int(o%3)
            
            ti = int(tile/3)
            tj = int(tile%3)
            # print(i, j, ti, tj)
            x_[:, i*s:(i+1)*s, j*s:(j+1)*s] = x[:, ti*s:(ti+1)*s, tj*s:(tj+1)*s] 
        return x_, y
        
def get_cifar_test(dataset, folder, batch_size, test=False):
    test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_transform_cifar_blur = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor(), transforms.Normalize(mean, std)])
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
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, pin_memory=True, num_workers = 8)
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
        # train_data = dset.CIFAR10(folder, train=True, transform=train_transform_cifar_, download=True)
        train_data = cifar10Nosiy(folder, train=True, transform=train_transform_cifar, nosiy_rate=0.2)
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
        def __init__(self, path, transform=None, exclude_cifar=False):
            data_file = open(path+'/data/300K_random_images.npy', "rb")                  
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
            index = (index + self.offset) % 300000

            if self.exclude_cifar:
                while self.in_cifar(index):
                    index = np.random.randint(300000)

            img = self.load_image(index)
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)

            return img, 0  # 0 is the class
        def __len__(self):
            return 300000

    ood_data = TinyImages(path, test_transform_cifar, False)
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
    
def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

class cifar10Nosiy(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, nosiy_rate=0.0, asym=False):
        super(cifar10Nosiy, self).__init__(root, transform=transform, target_transform=target_transform)
        if asym:
            # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            for s, t in zip(source_class, target_class):
                cls_idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(nosiy_rate * cls_idx.shape[0])
                noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                for idx in noisy_sample_index:
                    self.targets[idx] = t
            return
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=10, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return
        
if __name__ == '__main__':
    pass