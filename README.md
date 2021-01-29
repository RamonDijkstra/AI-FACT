# AI-FACT
In the context of the course FACT in AI at the Universiteit van Amsterdam (UvA), we attempted to reproduce a paper on confidentiality in AI. The paper reproduced in this repository is *Interpretable Complex-Valued Neural Networks for Privacy Protection* by [Xiang et. al](https://arxiv.org/abs/1901.09546#:~:text=Interpretable%20Complex%2DValued%20Neural%20Networks%20for%20Privacy%20Protection,-Liyao%20Xiang%2C%20Haotian&text=Previous%20studies%20have%20found%20that,without%20too%20much%20accuracy%20degradation.). Their paper provides a framework in which part of the AI processing can be moved from the device to the cloud without the loss of confidentiality due to adversarial attacks. Our reproduced paper can be found in this github repository.

This repository contains a PyTorch Lightning implementation of the experiments from the paper by [Xiang et. al](https://arxiv.org/abs/1901.09546#:~:text=Interpretable%20Complex%2DValued%20Neural%20Networks%20for%20Privacy%20Protection,-Liyao%20Xiang%2C%20Haotian&text=Previous%20studies%20have%20found%20that,without%20too%20much%20accuracy%20degradation.). To run the experiments, please follow the instructions below.

## Content
To test the two main claims in the paper, we implemented the following models:

### Baseline and complex models
* LeNet
* ResNet-56-α
* ResNet-110-α
* VGG-16

### Attacker models
* Inversion attacker 2

## Prerequisites
* Anaconda. Available at: https://www.anaconda.com/distribution/

## Getting started
1. Download the CUB-200 2011 dataset in .tgz format from https://drive.google.com/file/d/1hbzc_P1FuxMkcabkgn9ZKinBwW683j45/view.
2. Move the CUB_200_2011.tgz file to the folder *./data/CUB_200_2011*. If this folder does not exist, create it first.
3. Download all the pre-trained models from https://drive.google.com/drive/folders/1LnzXdh_Fg5lbIyDqpmbqjoyAHKmdhvdB?usp=sharing.
4. Move the pre-trained models to the folder *./saved_models*. If this folder does not exist, create it first.
5. Open Anaconda prompt and clone this repository:
```bash
git clone https://github.com/Ramonprogramming/AI-FACT
```
6. Move to the directory:
```bash
cd AI-FACT
```
7. Create the environment:
```bash
conda env create -f environment.yml
```
8. Activate the environment:
```bash
conda activate FACT_AI
```
9. View the notebook with the experimental results:
```bash
jupyter notebook results.ipynb
```

## Training the models
Models can be trained using the *main_pl.py* file. The model and training can be customized by passing command line arguments. The following arguments are supported:
```bash
usage: main_pl.py [-h] [--model MODEL] [--dataset DATASET]
			   [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
			   [--epochs EPOCHS] [--k K] [--log_dir LOG_DIR]
			   [--load_dir LOAD_DIR] [--progress_bar] [--seed SEED]
				 [--no_early_stopping] [--lr LR]

optional arguments:
  -h, --help            	Show help message and exit.
  --model MODEL			Model to use. Options: ['LeNet', 'Complex_LeNet', 'ResNet-56', 'Complex_ResNet-56', 'ResNet-110', 'Complex_ResNet-110', 'VGG-16', 'Complex_VGG-16']. Default is 'Complex_LeNet'.
  --dataset DATASET		Dataset to use. Only certain combinations with models are working. Default is 'CIFAR-10'.
					(Complex_)LeNet - ['CIFAR-10', 'CIFAR-100']
					(Complex_)ResNet-56 - ['CIFAR-10', 'CIFAR-100']
					(Complex_)ResNet-110 - ['CIFAR-10', 'CIFAR-100']
					(Complex_)VGG-16 - ['CUB-200']
  --batch_size BATCH_SIZE	Batch size. Accepts int values. Default is 256.
  --num_workers NUM_WORKERS	Number of workers for the dataloader. Accepts int values. Default is 0 (truly deterministic).
  --epochs EPOCHS		Number of epochs used in training. Accepts int values Default is 10.
  --k K				Level of k-anonimity. K-1 fake features are used when training. Accepts int values. Default is 2.
  --log_dir LOG_DIR		Directory for the PyTorch Lightning logs. Accepts string values. Default is 'model_logs/'.
  --load_dir LOAD_DIR		Directory where the model you want to load is stored. Default is None.
  --progress_bar 		Show a statusbar on the training progress or not. Disabled by default.
  --seed SEED			Seed used for reproducability. Accepts int values. Default is 42.
  --no_early_stopping 		Disable early stopping using the convergence criteria. Enabled by default.
  --lr LR			Learning rate to use for the model. Accepts int or float values. Default is 3e-4.
```

## Training the attackers
Attackers can be trained using the *train_attacker.py* file. The model and training can be customized by passing command line arguments. The following arguments are supported:
```bash
usage: train_attacker.py [-h] [--gan_model GAN_MODEL] [--dataset DATASET]
			   [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS]
			   [--epochs EPOCHS] [--k K] [--log_dir LOG_DIR]
			   [--load_gan LOAD_GAN] [--progress_bar] [--seed SEED]
				 [--no_early_stopping] [--lr LR]

optional arguments:
  -h, --help            	Show help message and exit.
  --gan_model GAN_MODEL		GAN of which model to use. Options: ['Complex_LeNet', 'Complex_ResNet-56', 'Complex_ResNet-110', 'Complex_VGG-16']. Default is 'Complex_LeNet'.
  --dataset DATASET		Dataset to use. Only certain combinations with models are working. Default is 'CIFAR-10'.
					(Complex_)LeNet - ['CIFAR-10', 'CIFAR-100']
					(Complex_)ResNet-56 - ['CIFAR-10', 'CIFAR-100']
					(Complex_)ResNet-110 - ['CIFAR-10', 'CIFAR-100']
					(Complex_)VGG-16 - ['CUB-200']
  --batch_size BATCH_SIZE	Batch size. Accepts int values. Default is 64.
  --num_workers NUM_WORKERS	Number of workers for the dataloader. Accepts int values. Default is 0 (truly deterministic).
  --epochs EPOCHS		Number of epochs used in training. Accepts int values Default is 10.
  --k K				Level of k-anonimity. K-1 fake features are used when training. Accepts int values. Default is 2.
  --log_dir LOG_DIR		Directory for the PyTorch Lightning logs. Accepts string values. Default is 'attacker_logs/'.
  --load_gan LOAD_GAN		Directory where the model for the GAN is stored. Is required.
  --progress_bar 		Show a statusbar on the training progress or not. Disabled by default.
  --seed SEED			Seed used for reproducability. Accepts int values. Default is 42.
  --no_early_stopping 		Disable early stopping using the convergence criteria. Enabled by default.
  --lr LR			Learning rate to use for the model. Accepts int or float values. Default is 3e-4.
```

## Authors
* Luuk Kaandorp - luuk.kaandorp@student.uva.nl
* Ward Pennink - ward.pennink@student.uva.nl
* Ramon Dijkstra - ramon.dijkstra@student.uva.nl
* Reinier Bekkenutte - reinier.bekkenutte@student.uva.nl

## Acknowledgements
* Pytorch Lightning implementation and some of the models were developed using information available in the Deep Learning Course of the UvA (https://uvadlc.github.io/).
* The U-net implementation for the inversion attack model was adapted from https://amaarora.github.io/2020/09/13/unet.html.
