#!/usr/bin/env python
# coding: utf-8

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
from objLoader_trimesh import trimesh_load_obj
import scipy.sparse as sp
import torch 
import dill
import pickle
import torch.nn as nn
from random import shuffle

from misc_utils import *
from utils_distance import *

device = 'cuda'

# import shutil
# shutil.rmtree('data/faust2500/processed')

dataset_path = 'FAUST/'

gt_list_path = 'FAUST_GIH_gt/'

os.makedirs(gt_list_path)

mesh_list = os.listdir(dataset_path)

#GIH_list = np.zeros((len(mesh_list), 6890, 6890),dtype="float32")

for mesh_idx in range(len(mesh_list)):
    mesh_path = dataset_path+'tr_reg_'+str(mesh_idx).zfill(3)+'.obj'

    mesh=trimesh_load_obj(mesh_path)

    mesh_points=torch.from_numpy(mesh.vertices).float().cuda().unsqueeze(0)
    mesh_faces=torch.from_numpy(mesh.faces).long().cuda().unsqueeze(0)
    Dg_r, grad, div, W, S, C = distance_GIH(mesh_points, mesh_faces) 
    print(mesh_points.shape)         
    print(mesh_faces.shape)

    gt_path = gt_list_path+'tr_reg_'+str(mesh_idx).zfill(3)+'.pkl'
    with open(gt_path, "wb") as f:
        pickle.dump(Dg_r.cpu().detach().numpy(), f)

