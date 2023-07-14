"""Implementation of sample attack."""
import os
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
from attack_methods import DI,gkern
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
from dct import *
# from Normalize import Normalize
from loader import ImageNet
from torch.utils.data import DataLoader
import argparse
# import pretrainedmodels

from Normalize import Normalize, TfNormalize
from torch import nn
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )

# num_workers=0


parser = argparse.ArgumentParser()
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--input_dir', type=str, default='./dataset/images', help='Input directory with images.')
parser.add_argument('--model_dir', type=str, default='./models/', help='model directory.') 

parser.add_argument('--model_name', type=str, default='tf_inception_v3', help='source model name.') 
parser.add_argument('--output_dir', type=str, default='./outputs/fia/', help='Output directory with adversarial images.') # 结尾记得加 /


parser.add_argument('--p', type=float, default=0.9, help='mean.')
parser.add_argument("--batch_size", type=int, default=2, help="How many images process at one time.") # batchsize
parser.add_argument("--N", type=int, default=2, help="The number of Spectrum Transformations") # 原20
parser.add_argument('--layer_name', type=str, default='1.InceptionV3_InceptionV3_Mixed_5c_Branch_1_Conv2d_0b_1x1_Conv2D', help='name.') 
# '1.InceptionV3_InceptionV3_Mixed_5c_Branch_1_Conv2d_0b_1x1_Conv2D'   v3->v4: 49.8; v2_adv: 8.3
# '1.InceptionV3_InceptionV3_Mixed_5d_Branch_0_Conv2d_0a_1x1_Conv2D'   v3->v4: 71.4; v2_adv: 16.9
# '1.InceptionV3_InceptionV3_Mixed_6a_Branch_0_Conv2d_1a_1x1_Conv2D'   v3->v4: 71.3; v2_adv: 14.2
# '1.InceptionV3_InceptionV3_Mixed_6b_Branch_0_Conv2d_0a_1x1_Conv2D'   v3->v4: 75.0; v2_adv: 18.8
# '1.InceptionV3_InceptionV3_Mixed_6c_Branch_0_Conv2d_0a_1x1_Conv2D'   v3->v4: 72.4; v2_adv: 16.6
# '1.InceptionV3_InceptionV3_Mixed_6d_Branch_0_Conv2d_0a_1x1_Conv2D'   v3->v4: 65.6; v2_adv: 14.4


parser.add_argument('--mean', type=float, default=np.array([0.5, 0.5, 0.5]), help='mean.')
parser.add_argument('--std', type=float, default=np.array([0.5, 0.5, 0.5]), help='std.')
parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter_set", type=int, default=10, help="Number of iterations.") # 
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")


parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
parser.add_argument("--rho", type=float, default=0.5, help="Tuning factor")
parser.add_argument("--sigma", type=float, default=16.0, help="Std of random noise")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

transforms = T.Compose(
    [T.Resize(299), T.ToTensor()]
)

def normalize(grad,opt=2):
    if opt==0:
        nor_grad=grad
    elif opt==1: # 1范数：除以绝对值之和
        abs_sum=np.sum(np.abs(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/abs_sum
    elif opt==2:
        square = np.sum(np.square(grad),axis=(1,2,3),keepdims=True)
        nor_grad=grad/np.sqrt(square) # 开根号
    return nor_grad

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def save_image(images,names,output_dir):
    """save the adversarial images"""
    if os.path.exists(output_dir)==False:
        os.makedirs(output_dir)

    for i,name in enumerate(names):
        img = Image.fromarray(images[i].astype('uint8'))  # 直接从- [-1,1] 转到 uint8？
        img.save(output_dir + name)

T_kernel = gkern(7, 3) # 3,1,7,7

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        
        net.KitModel(model_path).eval().to(device))  ##  net.KitModel(model_path).eval().cuda(),)
    
    return model

def get_mid_output(m, i, o):
    global mid_outputs 
    mid_outputs = i[0]

def backward_hook(module, input_grad, output_grad):
    global mid_gds
    mid_gds = input_grad[0]
    
    
def fia(images, gt, model, min, max):
    """
    The attack algorithm of our proposed Spectrum Simulate Attack
    :param images: the input images
    :param gt: ground-truth
    :param model: substitute model
    :param mix: the mix the clip operation 
    :param max: the max the clip operation
    :return: the adversarial images
    """
    image_width = opt.image_width
    momentum = opt.momentum
    num_iter = 10
    eps = opt.max_epsilon / 255.0
    alpha = eps / num_iter
    x = images.clone() # clone
    grad = 0 # 动量项 
    rho = opt.rho
    N = opt.N
    sigma = opt.sigma


    ###### weight 
    w = 0
    for n in range(N):
        
        handlers = []
        for (name, module) in model.named_modules():
            if name == opt.layer_name:
                print(name)
                handlers.append(module.register_backward_hook(backward_hook))  
    
        mask = np.random.binomial(1, opt.p, size=(x.shape[0],x.shape[1],x.shape[2],x.shape[3]))
        mask = torch.from_numpy(mask).to(device) # 这里需要加上与后面 x 都在一个 device 上
        x_temp = x.clone()
        x_mask = torch.tensor(x_temp.detach() * mask.detach(), dtype=torch.float )
        x_mask = x_mask.to(device)
        # x_mask = V(x, requires_grad = True)
        x_mask = V(x_mask, requires_grad = True)
        
        
        model.zero_grad()  
        
        
        source_out = model(x_mask)            
        one_hot = F.one_hot(gt, num_classes=1001)
        label_logit = source_out[0] * one_hot
        target = torch.max(label_logit,dim=1)[0]
        summ = torch.sum(target)
        summ.backward()

        wgrad = mid_gds
        wgrad = wgrad / torch.norm(wgrad, p=2, dim=(1,2,3), keepdim=True) 
        w = w + wgrad.detach()
    w = -w
    
    
    
    for i in range(num_iter):

        # DI-FGSM https://arxiv.org/abs/1803.06978
        # output_v3 = model(DI(x_idct))

        # TI-FGSM https://arxiv.org/pdf/1904.02884.pdf
        # noise = F.conv2d(noise, T_kernel, bias=None, stride=1, padding=(3, 3), groups=3)

        
        ##### loss 
        handlers = []
        for (name, module) in model.named_modules():
            if name == opt.layer_name:
                print(name)
                handlers.append(module.register_forward_hook(get_mid_output))
          
        x = V(x.to(device), requires_grad=True)
        source_out = model(x)
        feature = mid_outputs    
        weight_f = feature * w
        loss = torch.sum(weight_f) / torch.numel(weight_f)
        loss.backward()
        
        
        ##### MI-FGSM https://arxiv.org/pdf/1710.06081.pdf
        
        # x = V(x, requires_grad = True)
        # output = model(x)
        # loss = F.cross_entropy(output[0], gt)
        # loss.backward()      
        noise = x.grad.data  # 当前噪声 
        
        noise = noise / torch.abs(noise).mean([1, 2, 3], keepdim=True)
        noise = momentum * grad + noise
        grad = noise

        x = x + alpha * torch.sign(noise)
        x = clip_by_tensor(x, min, max)
    return x.detach()

def main():

    ## model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
    ##                             pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().cuda())
    # model = torch.nn.Sequential(Normalize(opt.mean, opt.std),
    #                             pretrainedmodels.inceptionv3(num_classes=1000, pretrained='imagenet').eval().to(device))

    model = get_model(opt.model_name, opt.model_dir) # 图片进入模型之前，先进入正则化网络变到 [-1,1]


    X = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(X, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=0)  # 原8 

    for images, images_ID,  gt_cpu in tqdm(data_loader):

        ## gt = gt_cpu.cuda()
        ## images = images.cuda()
        gt = gt_cpu.to(device)
        images = images.to(device)              
        
        images_min = clip_by_tensor(images - opt.max_epsilon / 255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + opt.max_epsilon / 255.0, 0.0, 1.0)

        adv_img = fia(images, gt, model, images_min, images_max)
        adv_img_np = adv_img.cpu().numpy()
        adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
        save_image(adv_img_np, images_ID, opt.output_dir)

if __name__ == '__main__':
    main()