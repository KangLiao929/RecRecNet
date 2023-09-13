import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from model.network import build_model, MSNetwork
from dataset import *
import os
import numpy as np
import skimage
import cv2
import skimage.measure

def draw_mesh_on_warp(warp, f_local, grid_h, grid_w):
    
    height = warp.shape[0]
    width = warp.shape[1]
    
    min_w = np.minimum(np.min(f_local[:,:,0]), 0).astype(np.int32)
    max_w = np.maximum(np.max(f_local[:,:,0]), width).astype(np.int32)
    min_h = np.minimum(np.min(f_local[:,:,1]), 0).astype(np.int32)
    max_h = np.maximum(np.max(f_local[:,:,1]), height).astype(np.int32)
    cw = max_w - min_w
    ch = max_h - min_h
    
    pic = np.ones([ch+10, cw+10, 3], np.int32)*255
    pic[0-min_h+5:0-min_h+height+5, 0-min_w+5:0-min_w+width+5, :] = warp
    
    warp = pic
    f_local[:,:,0] = f_local[:,:,0] - min_w+5
    f_local[:,:,1] = f_local[:,:,1] - min_h+5
    
    point_color = (0, 255, 0)
    thickness = 2
    lineType = 8
    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):
            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
              
    return warp

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return

def test(args):
    
    path = os.path.dirname(os.path.abspath(__file__))

    IMG_DIR = os.path.join(path, 'results/', 'img/')

    MESH_DIR = os.path.join(path, 'results/', 'mesh/')

    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
    if not os.path.exists(MESH_DIR):
        os.makedirs(MESH_DIR)

    test_data = TestDataset(data_path=args.test_path)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)
    
    # define the network
    net = MSNetwork(args.grid_w, args.grid_h)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device('cuda:{}'.format(args.gpu))
        net = net.to(device)

    #load the existing models if it exists
    MODEL_DIR = args.model_path
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)
        
        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')
  
    print("##################start testing#######################")
    net.eval()
    for i, batch_value in enumerate(test_loader):

        input_tesnor = batch_value[0].float()
        gt_tesnor = batch_value[1].float()

        if torch.cuda.is_available():
            input_tesnor = input_tesnor.cuda()
            gt_tesnor = gt_tesnor.cuda()

        batch_out = build_model(net, input_tesnor, args.grid_w, args.grid_h)
        rectangling = batch_out['rectangling']
        ori_mesh = batch_out['ori_mesh']
    
        rectangling_np = ((rectangling[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        input_np = ((input_tesnor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        gt_np = ((gt_tesnor[0]+1)*127.5).cpu().detach().numpy().transpose(1,2,0)
        ori_mesh_np = ori_mesh[0].cpu().detach().numpy()
            
        path = IMG_DIR + str(i+1) + ".jpg"
        cv2.imwrite(path, rectangling_np)
        
        input_with_mesh = draw_mesh_on_warp(input_np, ori_mesh_np, args.grid_w, args.grid_h)
        path = MESH_DIR + str(i+1) + ".jpg"
        cv2.imwrite(path, input_with_mesh)
            
        torch.cuda.empty_cache()


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='test/')
    parser.add_argument('--model_path', type=str, default='model/')
    parser.add_argument('--grid_h', type=int, default=8)
    parser.add_argument('--grid_w', type=int, default=8)

    print('<==================== Testing ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)