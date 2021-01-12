# Dataloaders
This folder contains multiple dataloaders used for the experiments on multiple datasets. The following dataloaders are included:
* celeba_loader - loads the CelebA dataset
* cifar10_loader - loads the CIFAR-10 dataset
* cifar100_loader - loads the CIFAR-100 dataset

## Accepted arguments
All dataloaders accept the following arguments:
* batch_size - size of the batches that is returned by the dataloader. Default is 128.
* num_workers - number of workers used to retrieve the datasets. Default is 2.
