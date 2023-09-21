import os
import sys
import warnings

import streamlit as st

sys.path.append('./')
sys.path.append('/home/blli/SDflow')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import copy
import pickle
import random
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from common.loggerx import LoggerXBase
from common.ops import load_network
from models.attribute_estimator import AttributeClassifier
from models.e4e import Encoder4Editing
from models.editor import SDFlow
from models.face_align.dlib_face_align import face_alignment
from models.flows.flow import cnf
from models.stylegan2 import Generator

torch.autograd.set_grad_enabled(False)

st.set_page_config(
    page_title="SDflow Demo",
    page_icon="🚀",
    layout="wide",
)


class config(object):
    stygan2_weights='./data/stylegan2-ffhq-config-f.pt'
    e4e_weights='./data/e4e_ffhq_encode.pt'
    
    attribute_index= [15,20,39]
    attribute_names = ['EyeGlasses','Gender','Young']
    attribute_index2name = {index:name for index,name in zip(attribute_index,attribute_names)}
    attribute_name2index = {name:index for index,name in zip(attribute_index,attribute_names)}
    
    attribute_weights='./data/r34_a40_age_256_classifier.pth'

    img_size=256

args = config()

device = "cuda" if torch.cuda.is_available() else 'cpu'

def map_range(value,in_min=0,in_max=1,out_min=-1.2,out_max=0.2):
    return (value-in_min)*(out_max-out_min)/(in_max-in_min)+out_min

@torch.no_grad()
def decoder(latent,img_size=256):
    with torch.no_grad():
        img = G(
            [latent],
            input_is_latent=True,
            randomize_noise=False,
        )[0].clamp(min=-1, max=1)
    if img_size != 1024:
        img = F.interpolate(img, size=(img_size, img_size), mode='bilinear', align_corners=True)
    return img

@st.cache_resource()
def load_model():
    
    e4e_encoder = Encoder4Editing(num_layers=50, mode='ir_se', stylegan_size=1024, checkpoint_path=args.e4e_weights).cuda().eval()

    print('** e4e encoder loads successfully!')

    # use decoder image classifier
    attributes_model = AttributeClassifier(backbone='r34')
    attributes_model.load_state_dict(load_network(args.attribute_weights))
    attributes_model = attributes_model.cuda().eval()
    print('** Attribute classifier initialization success !')

    # Initialization for stylegan2 model
    ckpt = torch.load(args.stygan2_weights,map_location='cpu')

    latent_avg = ckpt['latent_avg'].cuda()

    G = Generator(size=1024,style_dim=512,n_mlp=8)
    G.load_state_dict(ckpt['g_ema'])
    G.cuda().eval()
    print('** StyleGAN2 model initialization success !')
    
    return G,e4e_encoder,latent_avg,attributes_model

def show_attrs(attrs: dict):
    data = []
    for k, v in attrs.items():
        if k not in args.attribute_names:
            continue
        data.append([k, v])
    df = pd.DataFrame(data, columns=['Attributes', 'Values'])
    st.dataframe(df, use_container_width=True)


def show_image(col, image, content, attributes=None):
    with col:
        st.image(image, use_column_width=True)
        st.success(content)
        if attributes is not None:
            show_attrs(attributes)
        buf = BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        st.download_button(label="Download image", data=buf.getvalue(), file_name="image.png", mime="image/png")

G,e4e_encoder,latent_avg,attributes_model = load_model()

# ------------------------ main body ------------------------------

if 'edit_strength' not in st.session_state:
    st.session_state['edit_strength'] = [0. for _ in args.attribute_index]
    

st.title("✨ SDflow Demo 🏜")
st.info(' Let me help edit faces for any of your images. 😉\n')

st.sidebar.markdown("# Edit settings")

st.session_state['edit_attribute'] = st.sidebar.selectbox('Which attribute to edit?', options = args.attribute_names,index=0)
st.session_state['edit_strength'] = st.sidebar.slider('Edit strength (0-1) :', min_value=0.,max_value=1., value=0.7,step=0.05)

image_path = st.file_uploader("Upload Image 🚀", type=["png", "jpg", "bmp", "jpeg"],key='image_path')


if image_path is None:
    st.warning('⚠ Please upload your Image! 😯')
else:
    with st.spinner("Working.. 💫"):
        print('read image...')
        image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        col1, col2, col3 = st.columns(3)
        # with col1:
        #     st.image(image,use_column_width=True)
            
        aligned_image, inv_M = face_alignment(image, output_size=256)
        
        print(aligned_image.shape,type(aligned_image))
        
        
        
        input_image = torch.from_numpy(aligned_image).permute((2, 0, 1)).cuda().unsqueeze(0).float()
        input_image = (input_image / 255. - 0.5) * 2
        

        out,source = attributes_model(input_image)
        out = torch.sigmoid(out)
        attrs = {}
        for idx,name in args.attribute_index2name.items():
            attrs[name] = out[0,idx].cpu().numpy()
            
        show_image(col1,aligned_image,content='Aligned Face!',attributes=attrs)
        
        latent = e4e_encoder(input_image)
        latent += latent_avg
        
        projected = decoder(latent,512)
        out_project,source_project = attributes_model(projected)
        out_project = torch.sigmoid(out_project)
        attrs_project = {}
        for idx,name in args.attribute_index2name.items():
            attrs_project[name] = out_project[0,idx].cpu().numpy()
        
        
        projected = np.array(to_pil_image(make_grid(projected*0.5+0.5)))
        show_image(col2,projected,content='Projected Face',attributes=attrs_project)
        
        attr_num = args.attribute_name2index[st.session_state['edit_attribute']]
        transformer = SDFlow(ckpt_dir='./data/ckpt', attr_num=attr_num,scale=map_range(st.session_state['edit_strength']),device='cuda')

        new_latent = transformer.transform(latent, source,input_image)

        edit_image = decoder(new_latent)
        
        out_preds,source_preds = attributes_model(edit_image)
        out_preds = torch.sigmoid(out_preds)
        attrs_preds = {}
        for idx,name in args.attribute_index2name.items():
            attrs_preds[name] = out_preds[0,idx].cpu().numpy()
        
        edit_image = np.array(to_pil_image(make_grid(edit_image*0.5+0.5)))
        
        show_image(col3,edit_image,content='Edit attribute **{}**'.format(st.session_state['edit_attribute']),attributes=attrs_preds)

st.markdown(
    "<br><hr><center>Made by <strong>Binglei Li</strong> ✨",
    unsafe_allow_html=True)

# st.markdown('**Github**: https://github.com/some')

