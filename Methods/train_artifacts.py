#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 18:22:59 2022

@author: nelab
"""

#%% 0. Configuration
import os
import wandb
import torch
from easydict import EasyDict as edict

PROJECT = 'MNIST_tutorial'
ENTITY = 'nelab'
ARTIFACT = 'artifacts' # folder name to download artifact from wandb
GPUS = 1

train_config = edict({"gpus":GPUS,
                      "epochs": 100,
                      "es_counter": 10}) # no early stopping: es_counter: None

model_config = edict({'learning_rate':1e-4,
                'batch_size': 128,
                'in_channel': 1,
                'n_channel': 64,
                'fc_channel': 512,
                'stride': 2,
                'activation':'ReLU',
                'optimizer':'Adam',
                'weight_decay':1e-5,
                'dropout': 0,
                'cutmix_alpha': 0})


model_name = 'C3D_light' # C3D_light
model_alias_name = 'baseline' # 'baseline'


#%% 1. Load dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


# set data transform
train_transform = transforms.Compose([
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

# set directory to download dataset
download_root = './Data' # directory to save dataset

# check dataset files
if 'MNIST' in os.listdir(download_root): down_trig = False
else: down_trig = True

# load dataset
train_dataset = MNIST(download_root, transform=train_transform, train=True, download=down_trig)
valid_dataset = MNIST(download_root, transform=test_transform, train=False, download=down_trig)
test_dataset = MNIST(download_root, transform=test_transform, train=False, download=down_trig)

# set dataloader
trn_loader = DataLoader(dataset=train_dataset, batch_size=model_config.batch_size, shuffle=False, pin_memory=True)
val_loader = DataLoader(dataset=valid_dataset, batch_size=model_config.batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=model_config.batch_size, shuffle=True)


#%% 2. Initialize model
from Models.CNN_2D import C3D_light

def build_model_and_log(config, model_name=model_name, alias_name=model_alias_name):
    with wandb.init(project=PROJECT, group='artifact_log', job_type='initialize', config=config, entity=ENTITY) as run:
        config = wandb.config
        
        model = eval(model_name)(**config)
        
        model_artifact = wandb.Artifact(
            model_name, type='model',
            description=model_name,
            metadata=dict(config))
        
        torch.save(model.state_dict(), "initialized_model.pth")
        model_artifact.add_file("initialized_model.pth")

        run.log_artifact(model_artifact, aliases=alias_name)
            
        del model
        

build_model_and_log(model_config, model_name=model_name, alias_name=model_alias_name) # Run Only at first time


#%% 3. Training
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import uuid
import shutil

def train_and_log(train_config, model_config=None, model_name=model_name, model_alias_name=model_alias_name):
    with wandb.init(project=PROJECT,
                    group=model_name,
                    tags=[model_alias_name],
                    job_type='train',
                    config=train_config,
                    entity=ENTITY) as run:
        
        config = wandb.config
        hash_id = uuid.uuid4().hex
        os.mkdir(hash_id)
        cpfname = 'trained_model' # checkpoint folder name
        
        model_artifact = run.use_artifact(
            model_name+':'+model_alias_name)
        model_dir = model_artifact.download(os.path.join(ARTIFACT, model_name))
        model_path = os.path.join(model_dir, "initialized_model.pth")
        
        model_config = model_artifact.metadata
        model = eval(model_name)(**model_config)
        model.load_state_dict(torch.load(model_path))
        config.update(model_config)
        
        wandb_logger = WandbLogger()
        
        checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath = hash_id,
                                                            filename = cpfname,
                                                          save_top_k=1, monitor='valid/acc', mode='max')
        if config.es_counter:
            early_stop_callback = EarlyStopping(monitor="valid/acc", patience=config.es_counter, mode='max', verbose=True)
            callbacks = [checkpoint_callback, early_stop_callback]
        else: callbacks = checkpoint_callback
        
        trainer = pl.Trainer(gpus = config.gpus,
                             auto_lr_find = False,
                             auto_scale_batch_size = False,
                             max_epochs = config.epochs,
                             precision=16 if config.gpus else 32,
                             logger=wandb_logger,
                             callbacks=callbacks)
        
        # # search optimal learning rate
        # lr_log = dict()
        # # Run learning rate finder
        # lr_finder = trainer.tuner.lr_find(model, datamodule=dm, num_training=100)
        # # Pick point based on plot, or get suggestion
        # lr_log['lr_optim'] = lr_finder.suggestion()
        # # update hparams of the model
        # model.hparams.lr = lr_log['lr_optim']
        # config.update(lr_log)
        # print('optimal learning rate:'+str(lr_log['lr_optim']))
        
        trainer.fit(model, trn_loader, val_loader)
        
        # log artifact for trained model
        model_artifact = wandb.Artifact(model_name, type='model',
            description = model_name,
            metadata=dict(model_config))
        
        if config.es_counter: 
            model = eval(model_name).load_from_checkpoint(os.path.join(hash_id, cpfname+'.ckpt'))
        
        torch.save(model.state_dict(), os.path.join(hash_id, cpfname+'.ckpt'))
        model_artifact.add_file(os.path.join(hash_id, cpfname+'.ckpt'))
        # wandb.save(cpfname+'.ckpt')
        run.log_artifact(model_artifact, aliases=['latest', '_'.join([model_name, 'trained'])])
            
        wandb.alert(title="Training Finished", text=model_name)
        shutil.rmtree(hash_id)
        del model
        
train_and_log(train_config, model_config=None, model_name=model_name, model_alias_name=model_alias_name)


#%% 4. Testing
model_alias_name = '_'.join([model_name, 'trained'])

def test_and_log(train_config, model_config=None, model_name=model_name, model_alias_name=model_alias_name):
    with wandb.init(project=PROJECT,
                    group=model_name,
                    tags=[model_alias_name],
                    job_type='test',
                    config=train_config,
                    entity=ENTITY) as run:
        
        config = wandb.config
        
        model_artifact = run.use_artifact(
            model_name+':'+model_alias_name)
        model_dir = model_artifact.download(os.path.join(ARTIFACT, model_name))
        model_path = os.path.join(model_dir, "trained_model.ckpt")
        
        model_config = model_artifact.metadata
        model = eval(model_name)(**model_config)
        model.load_state_dict(torch.load(model_path))
        config.update(model_config)
        
        wandb_logger = WandbLogger()
        
        trainer = pl.Trainer(gpus = config.gpus,
                             auto_lr_find = False,
                             auto_scale_batch_size = False,
                             max_epochs = config.epochs,
                             precision=16 if config.gpus else 32,
                             logger=wandb_logger)
        
        
        trainer.test(model, test_loader)
        
        wandb.alert(title="Testing Finished", text=model_name)
        del model
        
test_and_log(train_config, model_config=None, model_name=model_name, model_alias_name=model_alias_name)



#%% 5. Interpretation
model_alias_name = '_'.join([model_name, 'trained'])

from captum.attr import GuidedGradCam
from captum.attr import LRP
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule, IdentityRule


def interpretation_LRP(config, model_name=model_name, model_alias_name=model_alias_name, save_path=None):
    with wandb.init(project=PROJECT,
                    group=model_name,
                    tags=[model_alias_name],
                    job_type='interpretation',
                    config=config,
                    entity=ENTITY) as run:
        config = wandb.config
        
        model_artifact = run.use_artifact(
            model_name+':'+model_alias_name)
        model_dir = model_artifact.download(os.path.join(ARTIFACT, model_name))
        model_path = os.path.join(model_dir, "trained_model.ckpt")
        
        model_config = model_artifact.metadata
        model = eval(model_name)(**model_config)
        model.load_state_dict(torch.load(model_path))
        config.update(model_config)
        
        # delete unnecessary structures
        del model._modules['train_acc']
        del model._modules['valid_acc']
        del model._modules['test_acc']
        del model._modules['train_pc']
        del model._modules['valid_pc']
        del model._modules['test_pc']
        del model._modules['train_rc']
        del model._modules['valid_rc']
        del model._modules['test_rc']
        del model._modules['loss_module']
        # if config.dropout != 0: del model._modules['drop']
        
        columns = ['image', 'heatmap', 'guess', 'truth', 'method']
        test_table = wandb.Table(columns = columns)
        
        for ti, sample_batched in enumerate(test_loader):
            if ti == 1: break
        sample_batched = next(iter(test_loader))
        test_X = sample_batched[0]
        test_y = sample_batched[1]
        output = model(test_X)
        pred = torch.argmax(torch.log_softmax(output, dim=1), 1)
    
        for tri in range(0, len(test_X)):
            sample = test_X[tri].unsqueeze(1)
            if config.attribution_method == 'LRP_Epsilon':
                for i in range(0, len(model._modules['conv_layer'])):
                    model._modules['conv_layer'][i].rule = EpsilonRule()
                for i in range(0, len(model._modules['fc_module'])):
                    model._modules['fc_module'][i].rule = EpsilonRule()
                if config.dropout != 0: model._modules['drop'].rule = IdentityRule()
                lrp = LRP(model)
                attribution = lrp.attribute(sample, target=pred[tri])
            elif config.attribution_method == 'LRP_A1B0':
                for i in range(0, len(model._modules['conv_layer'])):
                    model._modules['conv_layer'][i].rule = Alpha1_Beta0_Rule()
                for i in range(0, len(model._modules['fc_module'])):
                    model._modules['fc_module'][i].rule = Alpha1_Beta0_Rule()
                if config.dropout != 0: model._modules['drop'].rule = IdentityRule()
                lrp = LRP(model)
                attribution = lrp.attribute(sample, target=pred[tri])
            elif config.attribution_method == 'LRP_Gamma':
                for i in range(0, len(model._modules['conv_layer'])):
                    model._modules['conv_layer'][i].rule = GammaRule()
                for i in range(0, len(model._modules['fc_module'])):
                    model._modules['fc_module'][i].rule = GammaRule()
                if config.dropout != 0: model._modules['drop'].rule = IdentityRule()
                lrp = LRP(model)
                attribution = lrp.attribute(sample, target=pred[tri])
            elif config.attribution_method == 'LRP_Composite':
                for i in range(0, 4):
                    model._modules['conv_layer'][i].rule = Alpha1_Beta0_Rule()
                for i in range(4, len(model._modules['conv_layer'])):
                    model._modules['conv_layer'][i].rule = EpsilonRule()
                for i in range(0, len(model._modules['fc_module'])):
                    model._modules['fc_module'][i].rule = EpsilonRule(epsilon=0)
                if config.dropout != 0: model._modules['drop'].rule = IdentityRule()
                lrp = LRP(model)
                attribution = lrp.attribute(sample, target=pred[tri])
            elif config.attribution_method == 'GuidedGradCam':
                guided_gc = GuidedGradCam(model, model._modules['conv_layer'][4])
                attribution = guided_gc.attribute(sample, target=pred[tri])
                
            Input = sample[0,0,:,:].detach().cpu().data.numpy()
            Rel = attribution[0,0,:,:].detach().cpu().data.numpy()
            test_table.add_data(wandb.Image(Input), wandb.Image(Rel), str(pred[tri].numpy()), str(test_y[tri].numpy()), config.attribution_method)
        
        wandb.log({"table_key": test_table})
        wandb.alert(title="Interpretation Finished", text=model_name)

train_config.attribution_method = 'GuidedGradCam'

interpretation_LRP(train_config, model_name=model_name, model_alias_name=model_alias_name)


