# Jigsaw training for improving OOD
Official Implementation of the **"Exploring the Use of Jigsaw Puzzles in Out-of-Distribution Detection (Submitted to Computer Vision and Image Understanding)"**. 

In this study, we introduce Cufit, a curriculum fine-tuning paradigm of vision foundation model for medical image classification under label noise, based on linear probing and adapter modules.

---
# Getting Started
## Environment Setup
   This code is tested under LINUX 20.04 and Python 3.8 environment, and the code requires following packages to be installed:
    
   - [Pytorch](https://pytorch.org/): Tested under 2.0.1 version of Pytorch-GPU.
   - [torchvision](https://pytorch.org/vision/stable/index.html): which will be installed along Pytorch. Tested under 0.15.2 version.
   - [scipy](https://www.scipy.org/): Tested under 1.4.1 version.
   - [scikit-learn](https://scikit-learn.org/stable/): Tested under 0.22.1 version.

## Dataset Preparation
   Some public datasets are required to be downloaded for running evaluation. Required dataset can be downloaded in following links:    
   - [HAM10000](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
   - [APTOS-2019](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)
   - CIFAR10 and 100 will be automatically downloaded.

### Config file need to be changed for your path to download. For example,
~~~
# conf/HAM10000.json
{
    "epoch" : "100",
    "id_dataset" : "./cifar10",   # Your path to HAM10000 dataset
    "batch_size" : 32,
    "save_path" : "./cifar10/",   # Your path to checkpoint
    "num_classes" : 7
}
~~~

---
## How to Run
### To train a model by our setting (i.e., ours) with DINOv2-small for HAM10000 using 40% noise rate.
~~~
python train_rein_ours_three_head.py --data 'ham10000' --noise_rate 0.4 --netsize s --gpu 'gpu-num' --save_path 'save_name'
~~~
for example,
~~~
python train_rein_ours_three_head.py --data ham10000 --noise_rate 0.4 --netsize s --gpu 0 --save_path dinov2s_rein_3module
~~~
Other baselines full-training, linear probing, and adapter only are also available.

# License
The source code of this repository is released only for academic use. See the [license](LICENSE) file for details.

# Acknowledgement
This work was partially supported by Institute of Information \& communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00951, Development of Uncertainty-Aware Agents Learning by Asking Questions) and by ICT R\&D program of MSIT/IITP[2020-0-00857, Development of Cloud Robot Intelligence Augmentation, Sharing and Framework Technology to Integrate and Enhance the Intelligence of Multiple Robots].

