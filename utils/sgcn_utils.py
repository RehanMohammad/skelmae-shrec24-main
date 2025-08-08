## libraries
import random
import torch
import numpy as np

import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# from mediapipe import solutions

import math
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import shutil
import os
from tqdm import tqdm
import glob
from model.sgcn import SGCNModel



def train_epoch(epoch, num_epochs, model, optimizer, dataloader, criterion):
    model.train()

    pbar = tqdm(dataloader, total=len(dataloader))
    total_loss = 0

    for (V, y) in pbar:
        V = V.to(args.device)
        y = y.to(args.device)
        print(V.shape)
        identity = get_sgcn_identity( V.shape, args.device)
        optimizer.zero_grad()
        pred, _, pred_spatial_adj, _ = model(V, identity)
        loss_train = criterion(pred, y)
        loss_train.backward()
        optimizer.step()
        total_loss += loss_train.item()
        pbar.set_description(f'[%.3g/%.3g] train loss. %.2f' % (epoch, num_epochs, total_loss/len(y)))
    scheduler.step()
    
    return total_loss
        

def validate_epoch(model, dataloader, criterion):
    acc = 0.0
    n = 0
    pred_labels, true_labels = [], []
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader))
        for (V, y) in pbar:
            V = V.to(args.device)
            y = y.to(args.device)
            true_labels.append(y[0].item())
            identity = get_sgcn_identity(V.shape, args.device)
            output, *_ = model(V, identity)
            loss_valid = criterion(output, y)
            acc += (output.argmax(dim=1) == y.flatten()).sum().item()
            n += len(y.flatten())
            total_loss += loss_valid.item()
            
            pred_labels.append(output.argmax(dim=1)[0].item())
            desc = '[VALID]> loss. %.2f > acc. %.2g%%' % (total_loss/len(y), (acc / n)*100)
            pbar.set_description(desc)
                
    return total_loss, true_labels, pred_labels