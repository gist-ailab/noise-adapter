"""NIH Chest X-ray Dataset"""
import torch.utils.data as data
import torch
import os

from PIL import Image

class NIHchestXray():
    def __init__(self, root_dir, train = True, transforms=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.class2label = {
            'Atelectasis':0,
            'Cardiomegaly':1,
            'Effusion':2,
            'Infiltration':3,
            'Mass':4,
            'Nodule':5,
            'Pneumonia':6,
            'Pneumothorax':7,
            'Consolidation':8,
            'Edema':9,
            'Emphysema':10,
            'Fibrosis':11, 
            'Pleural_Thickening':12,
            'Hernia':13
        }
        self.root_dir = root_dir
        self.transform = transforms
        self.samples = []        
        self.label_txt = os.path.join(root_dir, 'Data_Entry_2017.csv')
        self.imgpath_txt = os.path.join(root_dir, 'train_val_list.txt' if train else 'test_list.txt')

        with open(self.imgpath_txt, 'r') as f:
            imglist = f.readlines()
            for i, img in enumerate(imglist):
                imglist[i] = img.replace('\n', '')

        self.images = dict()
        for i in range(1, 13):
            self.images[i] = os.listdir(os.path.join(root_dir, 'images_{}'.format(str(i).zfill(3)), 'images'))
        # print(self.images)
            
        with open(self.label_txt, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                line = line.split(',')
                imgname = line[0]
                label = line[1]
                # print(imgname, label)
                if imgname in imglist:
                    for i in range(1, 13):
                        if imgname in self.images[i]:
                            n = i
                    imgpath = os.path.join(root_dir, 'images_{}/images'.format(str(n).zfill(3)), imgname)
                    label_tensor = torch.zeros(14)
                    
                    label = label.split('|')
                    for lb in label:
                        if not lb == 'No Finding':
                            lb = self.class2label[lb]
                            label_tensor[lb] = 1
                    self.samples.append([imgpath, label_tensor])

                
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        sample = Image.open(sample)
        sample = sample.convert("RGB")

        if self.transform:
            sample = self.transform(sample)

        return sample, label


if __name__ == '__main__':
    train_set = NIHchestXray('data/chestxray')
    test_set = NIHchestXray('data/chestxray', False)
    print(len(train_set), len(test_set))