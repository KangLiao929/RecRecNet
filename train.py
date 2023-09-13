import argparse
from re import A
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import cv2
from model.network import build_model, MSNetwork
from datetime import datetime
from dataset import TrainDataset
import glob
from model.loss import *
import torchvision.models as models

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):

    path = os.path.dirname(os.path.abspath(__file__))

    SUMMARY_DIR = os.path.join(path, 'summary/', args.method)
    writer = SummaryWriter(log_dir=SUMMARY_DIR)

    MODEL_DIR = os.path.join(path, 'checkpoint/', args.method)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
    
    # curriculum dataset
    dataset = [args.train_path0, args.train_path1, args.train_path2]
    train_loader_curri = []
    for i in range(3):
        train_loader_temp = DataLoader(dataset=TrainDataset(data_path=dataset[i]), batch_size=args.batch_size, 
                                                                num_workers=4, shuffle=True, drop_last=True)
        train_loader_curri.append(train_loader_temp)

    # define the network
    net = MSNetwork(args.grid_w, args.grid_h)
    vgg_model = models.vgg19(pretrained=True)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
        net = net.to(device)
        vgg_model = vgg_model.to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    #load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        glob_iter = checkpoint['glob_iter']
        scheduler.last_epoch = start_epoch
        print('load model from {}!'.format(model_path))
    else:
        start_epoch = 0
        glob_iter = 0
        print('training from stratch!')

    print_interval = 300
    lr_reset_flag1 = True
    lr_reset_flag2 = True

    for epoch in range(start_epoch, args.max_epoch):
    
        print("start epoch {}".format(epoch))
        net.train()
        total_loss_sigma = 0.
        appearance_loss_sigma = 0.
        perception_loss_sigma = 0.
        inter_grid_loss_sigma = 0.
        
        # curriculum 1
        train_loader = train_loader_curri[0]
        
        # curriculum 2
        if epoch > 30 and epoch <= 80:
            train_loader = train_loader_curri[1]
            if(lr_reset_flag1):
                # renew the learning rate at the begining of a new curriculum
                adjust_learning_rate(optimizer, 1e-4)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
                lr_reset_flag1 = False

        # curriculum 3
        if epoch > 80:
            train_loader = train_loader_curri[2]
            if(lr_reset_flag2):
                adjust_learning_rate(optimizer, 1e-4)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
                lr_reset_flag2 = False

        print(epoch, 'lr={:.6f}'.format(optimizer.state_dict()['param_groups'][0]['lr']))

        for i, batch_value in enumerate(train_loader):

            input_tesnor = batch_value[0].float()
            gt_tesnor = batch_value[1].float()

            if torch.cuda.is_available():
                input_tesnor = input_tesnor.cuda()
                gt_tesnor = gt_tesnor.cuda()

            optimizer.zero_grad()

            batch_out = build_model(net, input_tesnor, args.grid_w, args.grid_h)
            rectangling = batch_out['rectangling']
            ori_mesh = batch_out['ori_mesh']
            
            # cal loss
            appearance_loss = cal_appearance_loss(rectangling, gt_tesnor) * 1.
            perception_loss = cal_perception_loss(vgg_model, rectangling, gt_tesnor) * 1e-4
            inter_grid_loss = cal_inter_grid_loss(ori_mesh, args.grid_w, args.grid_h) * 1.
            total_loss = appearance_loss + perception_loss + inter_grid_loss
            
            total_loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()
            
            total_loss_sigma += total_loss.item()
            appearance_loss_sigma += appearance_loss.item()
            perception_loss_sigma += perception_loss.item()
            inter_grid_loss_sigma += inter_grid_loss.item()
            
            # print loss etc.
            if i % print_interval == 0 and i != 0:
                total_loss_average = total_loss_sigma / print_interval
                appearance_loss_average = appearance_loss_sigma/ print_interval
                perception_loss_average = perception_loss_sigma/ print_interval
                inter_grid_loss_average = inter_grid_loss_sigma/ print_interval
                
                total_loss_sigma = 0.
                appearance_loss_sigma = 0.
                perception_loss_sigma = 0.
                inter_grid_loss_sigma = 0.

                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}]/[{:0>3}] Total Loss: {:.4f}  Appearance Loss: {:.4f}  Perception Loss: {:.4f}  Inter-Grid Loss: {:.4f} lr={:.8f}".format(epoch + 1, args.max_epoch, i + 1, len(train_loader), total_loss_average, appearance_loss_average, perception_loss_average, inter_grid_loss_average, optimizer.state_dict()['param_groups'][0]['lr']))
                
                # visualization
                writer.add_image("input", (input_tesnor[0]+1.)/2., glob_iter)
                writer.add_image("rectangling", (rectangling[0]+1.)/2., glob_iter)
                writer.add_image("gt", (gt_tesnor[0]+1.)/2., glob_iter)
               
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], glob_iter)
                writer.add_scalar('total loss', total_loss_average, glob_iter)
                writer.add_scalar('appearance loss', appearance_loss_average, glob_iter)
                writer.add_scalar('perception loss', perception_loss_average, glob_iter)
                writer.add_scalar('inter-grid loss', inter_grid_loss_average, glob_iter)

            glob_iter += 1
            
        scheduler.step()
        if ((epoch+1) % 10 == 0 or (epoch+1)==args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(MODEL_DIR, filename)
            state = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1, "glob_iter": glob_iter}
            torch.save(state, model_save_path)
    print("################## end training #######################")


if __name__=="__main__":
    
    print('<==================== setting arguments ===================>\n')
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=260)
    parser.add_argument('--grid_h', type=int, default=8)
    parser.add_argument('--grid_w', type=int, default=8)
    parser.add_argument('--train_path0', type=str, default='/dataset/4dof_dataset/')
    parser.add_argument('--train_path1', type=str, default='/dataset/8dof_dataset/')
    parser.add_argument('--train_path2', type=str, default='/dataset/Rectangling/')
    parser.add_argument('--method', type=str, default='tps_curriculum')
    
    args = parser.parse_args()
    print(args)
    
    print('<==================== start training ===================>\n')
    train(args)