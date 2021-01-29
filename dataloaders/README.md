# Dataloaders
This folder contains multiple dataloaders used for the experiments on multiple datasets. The following dataloaders are included:
* cub2011_loader - Loads the CUB-200 2011 dataset.
* cifar10_loader - Loads the CIFAR-10 dataset.
* cifar100_loader - Loads the CIFAR-100 dataset.

## Accepted arguments
All dataloaders accept the following arguments:
* batch_size - Size of the batches that is returned by the dataloader. Default is 128.
* num_workers - Number of workers used to retrieve the datasets. Default is 0.
