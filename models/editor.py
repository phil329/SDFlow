import os

import torch
import torch.nn as nn

from models.attribute_estimator import AttributeEstimator
from models.flows.flow import cnf

class SDFlow(object):
    def __init__(self,ckpt_dir,attr_num,attr_list=[15,20,39],scale=1.0,device='cuda') -> None:
        self.ckpt_dir = ckpt_dir
        self.device = device
        self.attr_num = attr_num
        self.scale = scale
        
        self.class_indices = attr_list
        
        self.labeldist = AttributeEstimator(backbone='resnet34',attribute_dim=len(self.class_indices)).to(self.device)
        
        filename = os.path.join(ckpt_dir,'labeldist-0010000')
        
        if os.path.exists(filename):
            self.labeldist.load_state_dict(torch.load(filename,map_location='cpu'),strict=True)
            self.labeldist.eval()
        else:
            raise ValueError(f'No such styleflow weight file {filename}')
    
        
        self.flow = cnf(512,'512-512-512-512-512',len(self.class_indices),1).to(self.device)
        filename = os.path.join(ckpt_dir, 'prior-0010000')
        
        if os.path.exists(filename):
            self.flow.load_state_dict(torch.load(filename,map_location='cpu'),strict=True)
            self.flow.eval()
        else:
            raise ValueError(f'No such styleflow weight file {filename}')
        
    def samples(self,targets):
        batch = targets.shape[0]
        z = torch.randn(batch,18,512).to(self.device)
        new_styles = self.flow(z, targets, reverse=True)
        return new_styles
    
    def transform(self, inputs, sources, images=None):
        
        targets = sources.clone()
        
        targets[(sources[:, self.attr_num] == 1), self.attr_num] = -self.scale
        targets[(sources[:, self.attr_num] == 0), self.attr_num] = 1+self.scale
        
        sources = sources[:, self.class_indices]
        targets = targets[:, self.class_indices]

        label_dist = self.labeldist(images,inputs)
        z = self.flow(inputs, label_dist, torch.zeros(targets.size(0), inputs.size(1), 1).to(self.device))[0]
        
        new_label_dist = label_dist.clone()
        new_label_dist[:,self.class_indices.index(self.attr_num)]= targets[:,self.class_indices.index(self.attr_num)]
        new_styles = self.flow(z, new_label_dist, reverse=True)
        return new_styles