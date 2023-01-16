# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
import os
#import random
import argparse
#from pickle import FALSE
import numpy as np
import torch.utils.data
import pandas as pd
import torch
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter
#from minetorch import Miner
import glob
from torchsummary import summary
#from astropy.io import fits
from my_dataset import myDataset
from CNNmodel import CNNmodel as create_model
#from torchsummary import summary
#from swin_transformer import SwinTransformer as create_model
#from swin_transformerv2 import SwinTransformerV2 as create_model
from toymodel import Toymodel as create_model1
#from swin_mlp import SwinMLP as create_model
#import torchvision.models as models
#create_model=models.resnet18()
import time
import timm
import timm.optim
import timm.scheduler
from utils import  train_one_epoch, evaluate
import torch.onnx




#import glob
print(torch.__version__)

def main(args):
    torch.cuda.empty_cache()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    #tb_writer = SummaryWriter()

    #train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(
    #args.data_path)

    
   

    sadata = pd.read_csv("E:\\学习\\大三下\\paper\\尼古拉\\galaxy\\first_step\\data\\label\\train_label.csv")
    #sadata = pd.read_csv("D:\\newidea\\newgalaxy\\data\\label\\train_labels2.csv")
    #sadata = pd.read_csv("D:\\newidea\\test_data.csv")
   
    #sadata = pd.read_csv("D:\\newidea\\newgalaxy\\date\\train0\\labels.csv")
    a = sadata.values
    
    #a=a[:1001]
    
    #print(a[:,1:8])
    #path = "D:\\newidea\\newgalaxy\\data\\train1\\"
    path="E:\\学习\\大三下\\paper\\尼古拉\\galaxy\\first_step\\data\\train\\"
    test_path = "E:\\学习\\大三下\\paper\\尼古拉\\galaxy\\first_step\\data\\test\\"
    #path_list = os.listdir("D:\\newidea\\newgalaxy\\data\\train1\\")
    path_list = os.listdir(path)
    #path_list=path_list[:-1]
    path_list.sort(key=lambda x: int(x.split('.')[0]))
    train_dataset=myDataset(a,path_list,path)
    
    #path_list=path_list[:1001]
    #b = pd.read_csv("D:\\newidea\\newgalaxy\\data\\label\\test_label.csv")
    b=pd.read_csv("E:\\学习\\大三下\\paper\\尼古拉\\galaxy\\first_step\\data\\label\\test_label.csv")
    b = b.values
    test_data = os.listdir(test_path)
    test_data.sort(key=lambda x: int(x.split('.')[0]))
    test_dataset=myDataset(b,test_data, test_path)
    #sadata = pd.read_csv("D:\\newidea\\newgalaxy\\data\\test\\labels.csv")
    # = sadata.values
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0,
              8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    prefetch_factor=2
    
    )
    time.sleep(1)
    test_loader=torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
        prefetch_factor=2,
        )

    #val_loader = torch.utils.data.DataLoader(val_dataset,
    #batch_size=batch_size,
    #shuffle=False,
    #pin_memory=True,
    #num_workers=nw,
    #collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)
    summary(model,[[1,64,64],[1,25,25]])
    onnx_path = "onnx_model_cnn.onnx"
    onnx_path1 = "onnx_model_swin.onnx"
    a=torch.randn((1,1,64,64)).to(device)
    b=torch.randn((1,1,25,25)).to(device)
    model1=create_model1(num_classes=args.num_classes).to(device)
    torch.onnx.export(model, (a,b), onnx_path)
    torch.onnx.export(model1, (a,b), onnx_path1)
    #summary(model)
    if args.weights != "":
        assert os.path.exists(
            args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除有关分类类别的权重
        #for k in list(weights_dict.keys()):
        #if "head" in k:
        #del weights_dict[k]
        model.load_state_dict(weights_dict, strict=False)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0)#定义优化器
    scheduler=timm.scheduler.CosineLRScheduler(optimizer=optimizer,#按照epoch调整学习率
                                               t_initial=args.epochs,
                                               lr_min=1e-8,
                                               warmup_t=10,
                                               warmup_lr_init=1e-8
                                               )
    for epoch in range(args.epochs):
        #train
        
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     
                                     device=device,
                                     path=path,
                                     epoch=epoch,                       
                                     batch_size=batch_size)
        
        scheduler.step(epoch)#学习率的调整
        tags = [
            "train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"
        ]
        
        # validate
        #if epoch%10==0:
        result = evaluate(model=model,
                                             data_loader=test_loader,
                                             device=device,
                                             path=path,
                                             epoch=epoch)
        result = np.array(result)
        if epoch%10==0:
            torch.save(model.state_dict(), "./weights_cnn_witpo/model-{}.pth".format(epoch))#保存模型
        np.savetxt(".\\test_predict_cnn_without_pooling\\predict" + str(epoch) + ".txt",
                   result.reshape(-1, 5),
                   fmt='%.8f')
        #保存文件的方式
        #tb_writer.add_scalar(tags[0], train_loss, epoch)
        #tb_writer.add_scalar(tags[1], train_acc, epoch)
        #tb_writer.add_scalar(tags[2], val_loss, epoch)
        #tb_writer.add_scalar(tags[3], val_acc, epoch)
        #tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

    #print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=201)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1E-4)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    #parser.add_argument('--data-path', type=str, default="D:\newidea\newgalaxy")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument(
        '--weights',
        type=str,
        default='')
        #'D:\\newidea\\newgalaxy\\core\\possible\\train50ep_lr1e-3_trans.pth'
        #D:\\newidea\\newgalaxy\\core\\weights\\model-20.pth
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device',
                        default='cuda:0',
                        help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
