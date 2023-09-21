import argparse
import os
import torch
import tqdm

import numpy as np
import os.path as osp
from torchvision.datasets.folder import is_image_file
from models.e4e import Encoder4Editing

torch.set_grad_enabled(False)

def load_e4e_model(checkpoint_path,device='cuda'):
    net = Encoder4Editing(num_layers=50, mode='ir_se', stylegan_size=1024, checkpoint_path=checkpoint_path).to(device)

    ckpt = torch.load(checkpoint_path,map_location='cpu')
    latent_avg = ckpt['latent_avg'].to(device)
    opts = ckpt['opts']
    
    print('** e4e encoder loads successfully!')
    print('** opts : ',opts)
    
    return net, latent_avg, opts


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--face_size', default=256, type=int, help='image output size')
    args = parser.parse_args()
    
    
    img = torch.randn((4,3,256,256)).cuda()
    net, latent_avg, opts = load_e4e_model('/home/blli/NoiseAging/pretrained_models/e4e_ffhq_encode.pt',device='cuda')
    
    out = net(img)+latent_avg
    
    print(out.shape)
    
    
