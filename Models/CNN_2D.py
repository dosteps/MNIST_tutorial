#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:41:57 2022

@author: nelab
"""

import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics

class C3D_light(pl.LightningModule):
    def __init__(self,
                 learning_rate=1e-4,
                 batch_size = 128,
                 in_channel = 1,
                 n_channel = 64,
                 fc_channel = 512,         
                 stride = 2,
                 activation='LeakyReLU',
                 optimizer='AdamW',
                 dropout=0,
                 weight_decay=1e-5,
                 num_classes=10,
                 num_fold=None,
                 cutmix_alpha = None):
        super().__init__()
        
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.train_pc = torchmetrics.Precision(average='macro', num_classes=num_classes, multiclass=True)
        self.valid_pc = torchmetrics.Precision(average='macro', num_classes=num_classes, multiclass=True)
        self.test_pc = torchmetrics.Precision(average='macro', num_classes=num_classes, multiclass=True)
        self.train_rc = torchmetrics.Recall(average='macro', num_classes=num_classes, multiclass=True)
        self.valid_rc = torchmetrics.Recall(average='macro', num_classes=num_classes, multiclass=True)
        self.test_rc = torchmetrics.Recall(average='macro', num_classes=num_classes, multiclass=True)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_module = nn.CrossEntropyLoss()
        self.save_hyperparameters()
        
        self.conv_layer = self._conv_layer_set(in_channel, n_channel, activation)
        fc1 = nn.Linear(256*4*4, fc_channel)
        fc2 = nn.Linear(fc_channel, fc_channel)
        fc3 = nn.Linear(fc_channel, num_classes)
        self.dropout = dropout
        self.drop=nn.Dropout(p=dropout)
        self.optimizer=optimizer
        self.weight_decay=weight_decay
        
        self.fc_module = nn.Sequential(
            fc1,
            getattr(nn, activation)(),
            fc2,
            getattr(nn, activation)(),
            fc3
        )
        
    def _conv_layer_set(self, in_c, n_c, activation):
        conv_layer = nn.Sequential(
        nn.Conv2d(in_c, n_c, kernel_size=(3, 3), stride=(1, 1), padding=0),
        nn.BatchNorm2d(n_c, track_running_stats = True),
        getattr(nn, activation)(),
        nn.Conv2d(n_c, n_c*2, kernel_size=(3, 3), stride=(1, 1), padding=0),
        nn.BatchNorm2d(n_c*2, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(n_c*2, n_c*4, kernel_size=(3, 3), stride=(1, 1), padding=0),
        getattr(nn, activation)(),
        nn.Conv2d(n_c*4, n_c*4, kernel_size=(3, 3), stride=(1, 1), padding=0),
        nn.BatchNorm2d(n_c*4, track_running_stats = True),
        getattr(nn, activation)(),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        return conv_layer
    
    def forward(self, x):
        # Set 1
        out = self.conv_layer(x)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        if self.dropout: out = self.drop(out)
        out = self.fc_module(out)
        return out
    
    def training_step(self, train_batch, batch_idx):
        x = train_batch[0]
        y = train_batch[1]
        y_hat = self(x)
        pred = torch.log_softmax(y_hat, dim=1)
        loss = self.loss_module(y_hat, y)
        #update and log
        self.train_acc(pred, y)
        self.train_pc(pred, y)
        self.train_rc(pred, y)
        self.log('train/loss', loss, on_step=True, on_epoch=False, sync_dist=False)
        self.log('train/acc', self.train_acc, on_step=True, on_epoch=False, sync_dist=False)
        self.log('train/precision', self.train_pc, on_step=True, on_epoch=False, sync_dist=False)
        self.log('train/recall', self.train_rc, on_step=True, on_epoch=False, sync_dist=False)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x = val_batch[0]
        y = val_batch[1]
        y_hat = self(x)
        pred = torch.log_softmax(y_hat, dim=1)
        val_loss = self.loss_module(y_hat, y)
        #update and log
        self.valid_acc(pred, y)
        self.valid_pc(pred, y)
        self.valid_rc(pred, y)
        self.log('valid/loss', val_loss, on_step=False, on_epoch=True, sync_dist=False)
        self.log('valid/acc', self.valid_acc, on_step=False, on_epoch=True, sync_dist=False)
        self.log('valid/precision', self.valid_pc, on_step=False, on_epoch=True, sync_dist=False)
        self.log('valid/recall', self.valid_rc, on_step=False, on_epoch=True, sync_dist=False)
            
    def test_step(self, test_batch, batch_idx):
        x = test_batch[0]
        y = test_batch[1]
        logits = self(x)
        return {'y': y.detach(), 'y_hat': logits.detach()}
    
    def test_epoch_end(self, outputs):  # 한 에폭이 끝났을 때 실행
        if len(outputs[0]['y'].shape)==0:
            y = torch.cat([x['y'].unsqueeze(-1) for x in outputs], dim=0)                
            y_hat = torch.cat([x['y_hat'].unsqueeze(-1) for x in outputs], dim=0)
        else:
            y = torch.cat([x['y'] for x in outputs], dim=0)           
            y_hat = torch.cat([x['y_hat'] for x in outputs], dim=0)
        pred = torch.log_softmax(y_hat, dim=1)
        avg_loss = self.loss_module(y_hat, y)
        self.test_acc(pred, y)
        self.test_pc(pred, y)
        self.test_rc(pred, y)
        # print(f"Epoch {self.current_epoch} acc:{acc} auc:{auc}")
        self.log('test/loss', avg_loss,  prog_bar=False, sync_dist=False)
        self.log('test/acc', self.test_acc,  prog_bar=False, sync_dist=False)
        self.log('test/precision', self.test_pc, on_step=False, on_epoch=True, sync_dist=False)
        self.log('test/recall', self.test_rc, on_step=False, on_epoch=True, sync_dist=False)
    
    def configure_optimizers(self):
      optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
      return optimizer


#%% identifying model structure

# from torchsummary import summary # custom func
# model_sm = C3D_light(None, None)
# summary(model_sm, input_size=(1, 28, 28), device="cpu")
