import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def get_vgg19_FeatureMap(vgg_model, input_255, layer_index):
    
    vgg_mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape((1,3,1,1))
    if torch.cuda.is_available():
        vgg_mean = vgg_mean.cuda()
    vgg_input = input_255-vgg_mean

    for i in range(0,layer_index+1):
        if i == 0:
            x = vgg_model.features[0](vgg_input)
        else:
            x = vgg_model.features[i](x)
            
    return x

def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))
        
def cal_appearance_loss(rectangling, gt):
    
    pixel_loss = l_num_loss(rectangling, gt, 1)
    
    return pixel_loss

def cal_perception_loss(vgg_model, rectangling, gt):
    
    rectangling_feature = get_vgg19_FeatureMap(vgg_model, (rectangling+1)*127.5, 24)
    gt_feature = get_vgg19_FeatureMap(vgg_model, (gt+1)*127.5, 24)
    
    feature_loss = l_num_loss(rectangling_feature, gt_feature, 2)
    
    return feature_loss


class VGG19(torch.nn.Module):
  def __init__(self, resize_input=False):
    super(VGG19, self).__init__()
    features = models.vgg19(pretrained=True).features.cuda()

    self.resize_input = resize_input
    self.mean=torch.Tensor([0.485, 0.456, 0.406]).cuda()
    self.std=torch.Tensor([0.229, 0.224, 0.225]).cuda()
    prefix = [1,1, 2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5]
    posfix = [1,2, 1,2, 1,2,3,4, 1,2,3,4, 1,2,3,4]
    names = list(zip(prefix, posfix))
    self.relus = []
    for pre, pos in names:
      self.relus.append('relu{}_{}'.format(pre, pos))
      self.__setattr__('relu{}_{}'.format(pre, pos), torch.nn.Sequential())

    nums = [[0,1], [2,3], [4,5,6], [7,8],
     [9,10,11], [12,13], [14,15], [16,17],
     [18,19,20], [21,22], [23,24], [25,26],
     [27,28,29], [30,31], [32,33], [34,35]]

    for i, layer in enumerate(self.relus):
      for num in nums[i]:
        self.__getattr__(layer).add_module(str(num), features[num])

    for param in self.parameters():
      param.requires_grad = False

  def forward(self, x):
    x = (x+1)/2
    x = (x-self.mean.view(1,3,1,1)) / (self.std.view(1,3,1,1))
    if self.resize_input:
      x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True)
    features = []
    for layer in self.relus:
      x = self.__getattr__(layer)(x)
      features.append(x)
    out = {key: value for (key,value) in list(zip(self.relus, features))}
    return out


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss
    

def cal_intra_grid_loss(mesh, grid_w=8, grid_h=8):
    
    min_w = 256/grid_w /8
    mix_h = 256/grid_h /8
    
    delta_x = mesh[:,:,0:grid_w,0] - mesh[:,:,1:grid_w+1,0]
    delta_y = mesh[:,0:grid_h,:,1] - mesh[:,1:grid_h+1,:,1]
        
    loss_x = F.relu(delta_x + min_w)
    loss_y = F.relu(delta_y + mix_h)
        
    loss = torch.mean(loss_x) + torch.mean(loss_y)
        
    return loss    
    
def cal_inter_grid_loss(mesh, grid_w=8, grid_h=8):
    w_edges = mesh[:,:,0:grid_w,:] - mesh[:,:,1:grid_w+1,:]

    cos_w = torch.sum(w_edges[:,:,0:grid_w-1,:] * w_edges[:,:,1:grid_w,:],3) / (torch.sqrt(torch.sum(w_edges[:,:,0:grid_w-1,:]*w_edges[:,:,0:grid_w-1,:],3))*torch.sqrt(torch.sum(w_edges[:,:,1:grid_w,:]*w_edges[:,:,1:grid_w,:],3)))
    delta_w_angle = 1 - cos_w
    
    h_edges = mesh[:,0:grid_h,:,:] - mesh[:,1:grid_h+1,:,:]
    cos_h = torch.sum(h_edges[:,0:grid_h-1,:,:] * h_edges[:,1:grid_h,:,:],3) / (torch.sqrt(torch.sum(h_edges[:,0:grid_h-1,:,:]*h_edges[:,0:grid_h-1,:,:],3))*torch.sqrt(torch.sum(h_edges[:,1:grid_h,:,:]*h_edges[:,1:grid_h,:,:],3)))
    delta_h_angle = 1 - cos_h
    
    loss = torch.mean(delta_w_angle) + torch.mean(delta_h_angle)
    
    return loss  
    
    
def cal_A_one_point(point):
    # input: [bs, grid_h, grid_w, 2]
    # output: [bs, grid_h, grid_w, 2, 4]
    # |x|   ---->   |x -y 1 0|
    # |y|           |y  x 0 1|
    bs, gh, gw, _  = point.size()
    a2 = torch.stack([-1*point[...,1], point[...,0]],3).unsqueeze(4)
    a34 = torch.tensor([[1.,0.],[0.,1.]]).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bs, gh, gw, -1, -1)
    if torch.cuda.is_available():
        a2 = a2.cuda()
        a34 = a34.cuda()
    A_one_point = torch.cat([point.unsqueeze(4), a2, a34], 4)
    return A_one_point
    
    
def cal_distortion_term(init_mesh, target_mesh, overlap, grid_w=8, grid_h=8):
    #  input:
    #  init_mesh [bs, grid_h+1, grid_w+1, 2]
    #  target_mesh [bs, grid_h+1, grid_w+1, 2]
    #  output:
    #  distortion loss
    
    init_left_top = init_mesh[:,0:grid_h, 0:grid_w,:]
    init_left_bottom = init_mesh[:,1:grid_h+1, 0:grid_w,:]
    init_right_top = init_mesh[:,0:grid_h, 1:grid_w+1,:]
    init_right_bottom = init_mesh[:,1:grid_h+1, 1:grid_w+1,:]
    
    target_left_top = target_mesh[:,0:grid_h, 0:grid_w,:]
    target_left_bottom = target_mesh[:,1:grid_h+1, 0:grid_w,:]
    target_right_top = target_mesh[:,0:grid_h, 1:grid_w+1,:]
    target_right_bottom = target_mesh[:,1:grid_h+1, 1:grid_w+1,:]
    
    A_left_top = cal_A_one_point(init_left_top)
    A_left_bottom = cal_A_one_point(init_left_bottom)
    A_right_bottom = cal_A_one_point(init_right_bottom)
    A_right_bottom = cal_A_one_point(init_right_bottom)
    A = torch.cat([A_left_top, A_left_bottom, A_right_bottom, A_right_bottom], 3)
    A = torch.reshape(A, [-1,8, 4])
    
    V = torch.cat([target_left_top, target_left_bottom, target_right_top, target_right_bottom], 3)
    V = torch.reshape(V, [-1,8,1])
    
    error = torch.matmul(torch.matmul(torch.matmul(A, torch.inverse(torch.matmul(A.permute(0,2,1), A))), A.permute(0,2,1)), V) - V
    error = torch.reshape(error, [-1, grid_h, grid_w, 8])
    error = torch.mean(error**2, dim=3)
    error = error*overlap
    
    return torch.mean(error)
    
    


    
    