import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from types import SimpleNamespace

# PyTorch Lightning
import pytorch_lightning as pl

# Callbacks 
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from models.resnet.resnet import *

from dataloaders.cifar10_loader import load_data as load_cifar10_data
from dataloaders.cifar100_loader import load_data as load_cifar100_data

# initialize our dataset dictionary
dataset_dict = {}
dataset_dict['CIFAR-10'] = load_cifar10_data
dataset_dict['CIFAR-100'] = load_cifar100_data

class CIFARTrainer(pl.LightningModule):
    
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        # Create model
        self.model = create_model(model_name, model_hparams)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()
        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    
    
    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)
    
    
    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, "Unknown optimizer: \"%s\"" % self.hparams.optimizer_name
        
        # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        return [optimizer], [scheduler]
    
    
    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader. 
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        
        self.log('train_acc', acc, on_step=False, on_epoch=True) # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log('train_loss', loss)
        return loss # Return tensor to call ".backward" on
    
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('val_acc', acc) # By default logs it per epoch (weighted average over batches)
        
        
    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log('test_acc', acc) # By default logs it per epoch (weighted average over batches), and returns it afterwards

# Setting the seed
pl.seed_everything(42)

#Set device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

model_dict = {}

def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, "Unknown model name \"%s\". Available models are: %s" % (model_name, str(model_dict.keys()))

model_dict["ResNet"] = ResNet

act_fn_by_name = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "gelu": nn.GELU
}

CHECKPOINT_PATH = 'complex_logs/'

def train_model(model_name, save_name=None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name

    # # load the data from the dataloader  
    # classes, trainloader, testloader = load_data('CIFAR-10', 256, 0)

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4)
        
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),                                  # Where to save models
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"), # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
                         gpus=1 if str(device)=="cuda:0" else 0,                                                     # We run on a single GPU (if possible)
                         max_epochs=180,                                                                             # How many epochs to train for if no patience is set
                         callbacks=[LearningRateMonitor("epoch")],                                                   # Log learning rate every epoch
                         progress_bar_refresh_rate=1)                                                                # In case your notebook crashes due to the progress bar, consider increasing the refresh rate
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
    
    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        model = CIFARTrainer.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(42) # To be reproducable
        model = CIFARTrainer(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = CIFARTrainer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training
        
    # Test best model on validation and test set
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    
    return model, result

def load_data(dataset='CIFAR-10', batch_size=256, num_workers=0):
    """
    Function for loading a dataset based on the given command line arguments.
    
    Inputs:
        dataset - String indicating the dataset to use. Default = 'CIFAR-10'
        batch_size - Int indicating the size of the mini batches. Default = 256
        num_workers - Int indicating the number of workers to use in the dataloader. Default = 0 (truly deterministic)
    """
    
    # load the dataset if possible
    if dataset in dataset_dict:
        return dataset_dict[dataset](batch_size, num_workers)
    # alert the user if the given dataset does not exist
    else:
        assert False, "Unknown dataset name \"%s\". Available datasets are: %s" % (dataset, str(dataset_dict.keys()))

if __name__ == "__main__":
    resnet_model, resnet_results = train_model(model_name="ResNet", 
                                           model_hparams={"num_classes": 10,
                                                          "c_hidden": [16,32,64],
                                                          "num_blocks": [3,3,3],
                                                          "act_fn_name": "relu"}, 
                                           optimizer_name="SGD",
                                           optimizer_hparams={"lr": 0.1,
                                                              "momentum": 0.9,
                                                              "weight_decay": 1e-4})