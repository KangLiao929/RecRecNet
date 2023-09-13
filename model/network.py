import torch
import torch.nn as nn
import utils.torch_tps_transform as torch_tps_transform
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models
import torchvision.transforms as T

def get_rigid_mesh(batch_size, height, width, grid_w, grid_h):
    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()
    
    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2)
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt
    
def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) 
    
    return norm_mesh.reshape([batch_size, -1, 2]) 
    
    
def build_model(net, input_tensor, grid_w, grid_h):
    batch_size, _, img_h, img_w = input_tensor.size()
    
    offset = net(input_tensor)

    mesh_motion = offset.reshape(-1, grid_h+1, grid_w+1, 2)
        
    
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w, grid_w, grid_h)
    ori_mesh = rigid_mesh + mesh_motion

    
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_ori_mesh = get_norm_mesh(ori_mesh, img_h, img_w)
    
    output_tps = torch_tps_transform.transformer(input_tensor, norm_rigid_mesh, norm_ori_mesh, (img_h, img_w))
    
    out_dict = {}
    out_dict.update(rectangling = output_tps, ori_mesh = ori_mesh, rigid_mesh = rigid_mesh)
    
        
    return out_dict

    
def get_res50_FeatureMap(resnet50_model):
    layers_list = []
    
    layers_list.append(resnet50_model.conv1)
    layers_list.append(resnet50_model.bn1)
    layers_list.append(resnet50_model.relu)
    layers_list.append(resnet50_model.maxpool)
    layers_list.append(resnet50_model.layer1)
    layers_list.append(resnet50_model.layer2)
    layers_list.append(resnet50_model.layer3)
    
    feature_extractor = nn.Sequential(*layers_list)
           
    return feature_extractor
    
class MSNetwork(nn.Module):

    def __init__(self, grid_w, grid_h):
        super(MSNetwork, self).__init__()
        
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.regressNet_part1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.regressNet_part2 = nn.Sequential(
            nn.Linear(in_features=8192, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),
            #nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=2048, out_features=(self.grid_h+1)*(self.grid_w+1)*2, bias=True)
        )
        
        # kaiming initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        resnet50_model = models.resnet.resnet50(pretrained=True)
        self.feature_extractor = get_res50_FeatureMap(resnet50_model)
        

    def forward(self, input_tesnor):
        batch_size, _, img_h, img_w = input_tesnor.size()
        feature = self.feature_extractor(input_tesnor)
        
        temp = self.regressNet_part1(feature)
        temp = temp.view(temp.size()[0], -1)
        offset = self.regressNet_part2(temp)
        
        return offset
        