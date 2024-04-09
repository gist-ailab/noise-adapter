# Copyright 2022 Cristóbal Alcázar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NIH Chest X-ray Dataset"""

import os
import datasets

from requests import get
from pandas import read_csv

class APTOS2019():
    def __init__(self, root_dir, train = True, transforms=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transforms
        dataset = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', data_dir='./data')
        dataset = dataset['train' if train else 'test']
            
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, label = self.samples[idx]
        sample = Image.open(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample, label


if __name__ == '__main__':
    from datasets import load_dataset

    dataset = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', data_dir='./data')
    print(dataset['train'])
    # print(dataset['test']['labels'])
