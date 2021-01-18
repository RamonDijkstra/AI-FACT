# AI-FACT
In the context of the course FACT in AI at the Universiteit van Amsterdam (UvA), we attempted to reproduce a paper on confidentiality in AI. The paper reproduced in this repository is *Interpretable Complex-Valued Neural Networks for Privacy Protection* by [Xiang et. al](https://arxiv.org/abs/1901.09546#:~:text=Interpretable%20Complex%2DValued%20Neural%20Networks%20for%20Privacy%20Protection,-Liyao%20Xiang%2C%20Haotian&text=Previous%20studies%20have%20found%20that,without%20too%20much%20accuracy%20degradation.). Their paper provides a framework in which part of the AI processing can be moved from the device to the cloud without the loss of confidentiality due to adversarial attacks. 

*TODO: stukje over wat er precies in de github staat*

## Prerequisites
* Anaconda. Available at: https://www.anaconda.com/distribution/

*TODO: lijst van prerequisites, waarschijnlijk alleen anaconda*

## Getting started
Open Anaconda prompt and clone this repository:
```bash
git clone https://github.com/Ramonprogramming/AI-FACT
```
Move to the directory:
```bash
cd AI-FACT
```
Create the environment:
```bash
conda env create -f environment.yml
```
Activate the environment:
```bash
conda activate FACT_AI
```
View the notebook with the experimental results:
```bash
jupyter notebook results.ipynb
```

## Running the experiments
New experiments can be conducted using the *main_pl.py* file. The model and training can be customized by passing command line arguments. The following arguments are supported:
```bash
usage: main_pl.py [-h] [--model MODEL] [--dataset DATASET]
			   [--batch_size BATCH_SIZE] [--num_workers NUM_WORKERS] 
			   [--epochs EPOCHS] [--k K] [--log_dir LOG_DIR] 
			   [--progress_bar] [--seed SEED] [--lr LR]

optional arguments:
  -h, --help            	Show help message and exit.
  --model MODEL				Model to use. Options: ['LeNet', 'ResNet']. Default is 'LeNet'.
  --dataset DATASET			Dataset to use. Only certain combinations with models are allowed. Default is 'CIFAR-10'.
							LeNet - ['CIFAR-10', 'CIFAR-100']
							ResNet - ['?', '?']
  --batch_size BATCH_SIZE	Batch size. Accepts int values. Default is 256.
  --num_workers NUM_WORKERS	Number of workers for the dataloader. Accepts int values. Default is 0 (truly deterministic). 
  --epochs EPOCHS			Number of epochs used in training. Accepts int values Default is 10.
  --k K						Level of k-anonimity. K-1 fake features are used when training. Accepts int values. Default is 2.
  --log_dir LOG_DIR			Directory for the PyTorch Lightning logs. Accepts string values. Default is 'complex_logs/'.
  --progress_bar 			Whether to show a statusbar on the training progress or not.
  --seed SEED				Seed used for reproducability. Accepts int values. Default is 42.
  --lr LR					Learning rate to use for the model. Accepts int or float values. Default is 3e-4.
```

## Authors
* Luuk Kaandorp - luuk.kaandorp@student.uva.nl
* Ward Pennink - ward.pennink@student.uva.nl
* Ramon Dijkstra - ramon.dijkstra@student.uva.nl
* Reinier Bekkenutte - reinier.bekkenutte@student.uva.nl

## Acknowledgements
* The U-net implementation for the inversion attack model was adapted from https://github.com/milesial/Pytorch-UNet.
*TODO: acknowledgements als die er zijn*
