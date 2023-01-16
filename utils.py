import os
import sys
import json
import pickle
import random
import numpy as np
from astropy.io import fits
import torch
from tqdm import tqdm

import torch.nn
import torch.optim as optim

#from sklearn.preprocessing import MinMaxScaler
#from torchvision import transforms
#from PIL import Image



def train_one_epoch(model, optimizer, data_loader, device, path, epoch,
                    batch_size):

    model.train()
    loss_function = torch.nn.HuberLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    optimizer.zero_grad()
    sample_num = 0
    #scaler = MinMaxScaler()
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images,psf,labels=data
        pred = model(images.to(device),psf.to(device))
        #pred_classes = torch.max(pred, dim=1)[1]
            #accu_num = 0
        loss = loss_function(pred.float(), labels.to(device).float())
        loss.backward()#反向传播
        accu_loss += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            #if (epoch + 1) % batch_size == 0:
        optimizer.step()#优化器优化
        optimizer.zero_grad()#梯度清零
        data_loader.desc = "[train epoch {}] loss: {:.6f}".format(
                epoch, loss)
            #del images, labels
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
    return accu_loss.item() / (step + 1)


@torch.no_grad()
def evaluate(model, data_loader, device, path, epoch):
    loss_function = torch.nn.SmoothL1Loss()
    result = []
    model.eval()

    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images,psf,labels=data
        pred = model(images.to(device),psf.to(device))
        mse=torch.nn.MSELoss()
        mse_loss=mse(pred.float(), labels.to(device).float())
        pred1 = pred.cpu().numpy()
        result.append(pred1)
        #pred_classes = torch.max(pred, dim=1)[1]
        #accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred.float(), labels.to(device).float())
        accu_loss += loss
        data_loader.desc = "[valid epoch {}] loss: {:.6f}, mse: {:.6f}".format(
        epoch,
        loss,
        mse_loss
       )

    return result


#accu_loss.item() / (step + 1), accu_num.item() / (sample_num +
#1), result
