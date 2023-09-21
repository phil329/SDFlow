import argparse
import os,sys

sys.path.insert(0,'/home/blli/SDflow/models/stylegan2')
sys.path.insert(0,'/home/blli/SDflow')

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn, optim
from torch.utils import data
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from common.loggerx import WANDBLoggerX
from common.id_loss import IDLoss
from common.ops import load_network
from models.dataset import SDFlowDataset
from models.flows.flow import cnf
from models.flows.utils import modify_one_attribute, standard_normal_logprob,normal_logprob
from models.attribute_estimator import AttributeClassifier,AttributeEstimator
from models.stylegan2.model import Generator
    

def generate_test_image(flow_model:torch.nn.Module,
                        stylegan2_model:torch.nn.Module,
                        test_label_dist:torch.Tensor,
                        ori_img:torch.Tensor,
                        origin_latent:torch.Tensor,
                        attributes:torch.Tensor,
                        mid_latent:torch.Tensor,
                        img_size=256):
    batchsize = ori_img.shape[0]
    
    #ori_img = F.interpolate(ori_img,(1024,1024))
    #img_ori = torchvision.utils.make_grid(ori_img,nrow=1,normalize=True,range=(-1,1))
    #img_recon = stylegan2_model().clamp(-1,1)
    img_recon = stylegan2_model([origin_latent.squeeze(1)],input_is_latent=True,randomize_noise=False)[0].clamp(-1, 1)
    img_recon = torchvision.utils.make_grid(img_recon,nrow=1,normalize=True,range=(-1,1))
    
    # images = [img,ori, img_recon]
    images = [img_recon]
    
    # generate attributes-change face one by one
    groups = attributes.shape[-1]
    zero_padding = torch.zeros((batchsize,18,1)).to(origin_latent)
    for i in tqdm(range(groups)):
        _,new_attributes = modify_one_attribute(attributes,idx=i)
        new_label_dist = test_label_dist.detach().clone()
        new_label_dist[:,i] = new_attributes[:,i]
        
        # print(mid_latent.shape,new_attributes.shape,zero_padding.shape)
        new_latents,_ = flow_model(mid_latent,new_label_dist,zero_padding,reverse=True)
        
        #tmp = stylegan2_model(new_latents).clamp(-1,1)
        tmp = stylegan2_model([new_latents],input_is_latent=True,randomize_noise=False)[0].clamp(-1, 1)
        tmp = torchvision.utils.make_grid(tmp,nrow=1,normalize=True,value_range=(-1,1))
        images.append(tmp)

    merge = torch.cat(images,dim=2)
    merge = to_pil_image(merge).resize((img_size*(groups+1),img_size*batchsize))
    return merge

'''
CUDA_VISIBLE_DEVICES=2 python train_sdflow.py --attribute_index 15 20 39 
'''

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="StyleFlow trainer")
    
    parser.add_argument("--latent_file",default='./data/ffhq_e4e_latents.pth', type=str, help="path to the latents")
    parser.add_argument("--preds_file",default='./data/ffhq_e4e_preds.pth', type=str, help="path to the preds")
    parser.add_argument('--index_file',default='./data/ffhq.txt',type=str)
    parser.add_argument('--image_root',default='/mnt/local1_ssd/blli/dataset/FFHQ/images256x256',type=str)
    
    # parameters for model structure
    parser.add_argument("--flow_modules", type=str, default='512-512-512-512-512')
    parser.add_argument("--attribute_index",nargs='*',default=[15,20,39], type=int, help="list of the face attributes index of CelebA")
    parser.add_argument("--stygan2_weights",default='./data/stylegan2-ffhq-config-f.pt',type=str,help='stylegan2 weights path')
    
    # parameters for save and name
    parser.add_argument("--model_name",default='SDFlow',type=str,help="model name")
    parser.add_argument('--run_name',default='default',type=str,help='this run name')
    parser.add_argument('--print_freq', type=int, default=10,help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1000,help='save frequency')
    
    
    # parameters for training 
    parser.add_argument("--img_size",type=int,default=256,help="image size for model")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # parameters for loss weight
    parser.add_argument("--nll_loss_weight", type=float, default=1)
    parser.add_argument("--reg_loss_weight", type=float, default=1)
    parser.add_argument('--kd_loss_weight',type=float,default=1)
    parser.add_argument('--mi_loss_weight',type=float,default=1)
    
    args = parser.parse_args()
    torch.manual_seed(0)
    
    logger = WANDBLoggerX(save_root=os.path.join('./output',args.model_name,args.run_name),
                          print_freq=args.print_freq,
                          config=args,
                          project='ProjectName',
                          entity='EntityName',
                          name='{}_{}'.format(args.model_name,args.run_name))
    
    attribute_index = torch.tensor(args.attribute_index,dtype=int)
    prior = cnf(512, args.flow_modules, len(args.attribute_index), 1)
    # prior.load_state_dict(torch.load('./pretrained_models/ffhq_prior.pth',map_location='cpu'),strict=True)
    
    train_transform = T.Compose([
        T.ToTensor(),
        T.Resize((args.img_size,args.img_size)),
        T.Normalize(mean=0.5,std=0.5)
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Resize((args.img_size,args.img_size)),
        T.Normalize(mean=0.5,std=0.5)
    ])
    train_dataset = SDFlowDataset(index_file=args.index_file,
                                         image_root=args.image_root,
                                         latents_file=args.latent_file,
                                         preds_file=args.preds_file,
                                         train=True,
                                         transform=train_transform)
    
    test_dataset = SDFlowDataset(index_file=args.index_file,
                                        image_root=args.image_root,
                                        latents_file=args.latent_file,
                                        preds_file=args.preds_file,
                                        train=False,
                                        transform=test_transform)
    
    train_loader = data.DataLoader(train_dataset, 
                                   shuffle=True, 
                                   batch_size=args.batch,
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   drop_last=True)
    test_batch = args.batch if args.batch<=8 else 8
    test_loader = data.DataLoader(test_dataset, 
                                   shuffle=False, 
                                   batch_size=test_batch,
                                   pin_memory=False,
                                   drop_last=True)
    print('** dataloader initialized successfully !')
    
    
    labeldist = AttributeEstimator(backbone='resnet50',attribute_dim=len(args.attribute_index)).cuda()
    
    optimizer = optim.Adam(list(prior.parameters())+list(labeldist.parameters()), lr=args.lr)
    
    logger.modules = [prior, labeldist, optimizer]
    
    # Initialization for stylegan2 model
    ckpt = torch.load(args.stygan2_weights,map_location='cpu')
    G = Generator(size=1024,style_dim=512,n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.cuda().eval()
    print('** StyleGAN2 model initialization success !')
    
    test_loader = iter(test_loader)
    test_img,test_latent,test_pred = next(test_loader)
    test_img,test_latent,test_pred = test_img.cuda(),test_latent.cuda(),test_pred.cuda()
    test_attributes = test_pred[:,attribute_index]
        
    for epoch in range(args.epochs):
        for i, datas in tqdm(enumerate(train_loader),total=len(train_loader)):
            n_iter = epoch*len(train_loader)+i
            
            img,latent,pred = datas
            img = img.cuda()
            latent = latent.cuda()
            pred = pred.cuda()
            attributes = pred[:,attribute_index]
            
            label_dist = labeldist(img,latent)
            kd_loss = F.mse_loss(label_dist, attributes)
            
            approx21, delta_log_p2 = prior(latent, label_dist, torch.zeros(args.batch, 18, 1).to(latent))
            
            # make base distribution standard normal distibution
            approx2 = standard_normal_logprob(approx21).view(args.batch, -1).sum(1, keepdim=True)
            delta_log_p2 = delta_log_p2.view(args.batch, 18, 1).sum(1)
            log_p2 = -(approx2 - delta_log_p2).mean() / (18*512)
            
            modify_idx, new_label_dist = modify_one_attribute(label_dist,mode='uniform')
            new_latents,_ = prior(approx21,new_label_dist,torch.zeros(args.batch, 18, 1).to(latent),reverse=True)
            
            new_face_tensors = G([new_latents],input_is_latent=True,randomize_noise=False)[0].clamp(-1, 1)
            new_face_tensors = F.interpolate(new_face_tensors, (args.img_size, args.img_size))
            recon_label_dist = labeldist(new_face_tensors,new_latents)
            
            mi_loss = F.mse_loss(recon_label_dist, new_label_dist)
            reg_loss = F.mse_loss(new_latents, latent)

            loss = args.kd_loss_weight * kd_loss +\
                args.nll_loss_weight * log_p2 +\
                args.reg_loss_weight * reg_loss +\
                args.mi_loss_weight * mi_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            if n_iter % args.save_freq==0:
                with torch.no_grad():
                    test_label_dist = labeldist(test_img,test_latent)
                    test_mid_latent, _ = prior(test_latent, test_label_dist, torch.zeros(test_batch, 18, 1).to(test_latent))
                    grid_img = generate_test_image(prior,G,test_label_dist, test_img,test_latent,test_attributes,test_mid_latent)
                    logger.save_image(grid_img,n_iter,'test')
                if n_iter % (args.save_freq)==0:
                    logger.checkpoints(n_iter)
                    
            logger.msg([kd_loss,log_p2,mi_loss,reg_loss],n_iter)
            