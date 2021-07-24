import torch.utils.data as data
import torch
import numpy as np
import trimesh
from utils_3d.objLoader_trimesh import trimesh_load_obj
import random
import os
import pickle

class SMPL_DATA(data.Dataset):
    def __init__(self, train=True,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./dataset_3d/SMPL/'
        self.sample_size = len(os.listdir(self.path))
        self.length = training_size
        #self.test_label_path = './datasets/FAUST/FAUST_list/seen_pose_list.txt'
        #self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):

        identity_mesh_i=np.random.randint(0,16)
        identity_mesh_p=np.random.randint(200,600)

        pose_mesh_i=np.random.randint(0,16)
        pose_mesh_p=np.random.randint(200,600)
        
        identity_mesh_path=self.path+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'.obj'
        pose_mesh_path=self.path+str(pose_mesh_i)+'_'+str(pose_mesh_p)+'.obj'
        gt_mesh_path=self.path+str(identity_mesh_i)+'_'+str(pose_mesh_p)+'.obj'
        
        
        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)
        gt_mesh=trimesh_load_obj(gt_mesh_path)

        
        
        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        pose_points = pose_mesh.vertices
        pose_faces=pose_mesh.faces
        gt_points = gt_mesh.vertices
        gt_faces=gt_mesh.faces
        # print(identity_mesh)
        # print('hi')

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        pose_points=pose_points-(pose_mesh.bbox[0]+pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))
        
        gt_points=gt_points-(pose_mesh.bbox[0]+pose_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        identity_faces=identity_faces
        pose_faces=pose_faces

        # print(pose_faces)
        # print(identity_faces)

        return pose_points, pose_faces, identity_points, identity_faces,gt_points


    def __len__(self):
        return self.length


class FAUST_DATA(data.Dataset):
    def __init__(self, train=True,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./dataset_3d/FAUST/'
        self.sample_size = len(os.listdir(self.path))
        self.length = training_size
        #self.test_label_path = './datasets/FAUST/FAUST_list/seen_pose_list.txt'
        #self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):

        identity_mesh_idx = np.random.randint(0,self.sample_size )

        pose_mesh_idx=np.random.randint(0,self.sample_size )

        identity_mesh_path = self.path+'tr_reg_'+str(identity_mesh_idx).zfill(3)+'.obj'
        pose_mesh_path = self.path+'tr_reg_'+str(pose_mesh_idx).zfill(3)+'.obj'

        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)

        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        pose_points = pose_mesh.vertices
        pose_faces=pose_mesh.faces

        # print(identity_mesh)
        # print('hi')

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        pose_points=pose_points-(pose_mesh.bbox[0]+pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        identity_faces=identity_faces
        pose_faces=pose_faces

        # print(pose_faces)
        # print(identity_faces)

        return pose_points, pose_faces, identity_points, identity_faces


    def __len__(self):
        return self.length


class FAUST_DATA_with_GIH(data.Dataset):
    def __init__(self, train=True,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./dataset_3d/FAUST/'
        self.path_GIH_gt='./dataset_3d/FAUST_GIH_gt/'
        self.sample_size = len(os.listdir(self.path))
        self.length = training_size
        #self.test_label_path = './datasets/FAUST/FAUST_list/seen_pose_list.txt'
        #self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):
        identity_mesh_idx=np.random.randint(0,self.sample_size )
        pose_mesh_idx=np.random.randint(0,self.sample_size )

        identity_mesh_path = self.path+'tr_reg_'+str(identity_mesh_idx).zfill(3)+'.obj'
        pose_mesh_path = self.path+'tr_reg_'+str(pose_mesh_idx).zfill(3)+'.obj'

        pose_mesh_path = self.path+'tr_reg_'+str(pose_mesh_idx).zfill(3)+'.obj'

        gt_path = self.path_GIH_gt+'tr_reg_'+str(identity_mesh_idx).zfill(3)+'.pkl'
        with open(gt_path, "rb") as f:
            GIH_gt = pickle.load(f)
        GIH_gt = torch.from_numpy(GIH_gt.astype(np.float32))


        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)

        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        pose_points = pose_mesh.vertices
        pose_faces=pose_mesh.faces

        # print(identity_mesh)
        # print('hi')

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        pose_points=pose_points-(pose_mesh.bbox[0]+pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        identity_faces=identity_faces
        pose_faces=pose_faces

        # print(pose_faces)
        # print(identity_faces)

        return pose_points, pose_faces, identity_points, identity_faces, GIH_gt


    def __len__(self):
        return self.length


class DFAUST_DATA(data.Dataset):
    def __init__(self, train=True,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.pose_path='./dataset_3d/DFAUST/scripts/dfaust/pose/'
        self.shape_path='./dataset_3d/DFAUST/scripts/dfaust/shape/'
        self.sample_size = len(os.listdir(self.pose_path))
        self.length = training_size
        #self.test_label_path = './datasets/FAUST/FAUST_list/seen_pose_list.txt'
        #self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):

        identity_mesh_idx = np.random.randint(0,self.sample_size )

        pose_mesh_idx=np.random.randint(0,self.sample_size )

        identity_mesh_path = self.shape_path+str(identity_mesh_idx).zfill(5)+'.obj'
        pose_mesh_path = self.pose_path+str(pose_mesh_idx).zfill(5)+'.obj'

        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)

        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        pose_points = pose_mesh.vertices
        pose_faces=pose_mesh.faces

        # print(identity_mesh)
        # print('hi')

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        pose_points=pose_points-(pose_mesh.bbox[0]+pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        identity_faces=identity_faces
        pose_faces=pose_faces

        # print(pose_faces)
        # print(identity_faces)

        return pose_points, pose_faces, identity_points, identity_faces


    def __len__(self):
        return self.length



class DFAUST_DATA_with_GIH(data.Dataset):
    def __init__(self, train=True,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.pose_path='./dataset_3d/DFAUST/scripts/dfaust/pose/'
        self.pose_path_gt='./dataset_3d/DFAUST/scripts/dfaust/pose_gt/'
        self.shape_path='./dataset_3d/DFAUST/scripts/dfaust/shape/'
        self.shape_path_gt='./dataset_3d/DFAUST/scripts/dfaust/shape_gt/'
        self.sample_size = len(os.listdir(self.pose_path))
        self.length = training_size
        #self.test_label_path = './datasets/FAUST/FAUST_list/seen_pose_list.txt'
        #self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):
        identity_mesh_idx=np.random.randint(0,self.sample_size )
        pose_mesh_idx=np.random.randint(0,self.sample_size )

        identity_mesh_path = self.shape_path+str(identity_mesh_idx).zfill(5)+'.obj'
        pose_mesh_path = self.pose_path+str(pose_mesh_idx).zfill(5)+'.obj'

        gt_path = self.shape_path_gt+str(identity_mesh_idx).zfill(5)+'.pkl'
        with open(gt_path, "rb") as f:
            GIH_gt = pickle.load(f)
        GIH_gt = torch.from_numpy(GIH_gt.astype(np.float32))


        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)

        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces
        pose_points = pose_mesh.vertices
        pose_faces=pose_mesh.faces

        # print(identity_mesh)
        # print('hi')

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        pose_points=pose_points-(pose_mesh.bbox[0]+pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        identity_faces=identity_faces
        pose_faces=pose_faces

        # print(pose_faces)
        # print(identity_faces)

        return pose_points, pose_faces, identity_points, identity_faces, GIH_gt


    def __len__(self):
        return self.length




class FAUST_DATA_LAP(data.Dataset):
    def __init__(self, train=True,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./dataset_3d/FAUST/'
        self.sample_size = len(os.listdir(self.path))
        self.length = training_size
        #self.test_label_path = './datasets/FAUST/FAUST_list/seen_pose_list.txt'
        #self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):

        identity_mesh_idx = np.random.randint(0,self.sample_size )

        pose_mesh_idx=np.random.randint(0,self.sample_size )

        identity_mesh_path = self.path+'tr_reg_'+str(identity_mesh_idx).zfill(3)+'.obj'
        pose_mesh_path = self.path+'tr_reg_'+str(pose_mesh_idx).zfill(3)+'.obj'

        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)

        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces

        pose_points = pose_mesh.vertices
        pose_faces=pose_mesh.faces

        pose_points_Lap=pose_mesh.Lap_vertices
        pose_faces_Lap=pose_mesh.Lap_faces

        # print(identity_mesh)
        # print('hi')

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        pose_points=pose_points-(pose_mesh.bbox[0]+pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        pose_points_Lap=pose_points_Lap-(pose_mesh.Lap_bbox[0]+pose_mesh.Lap_bbox[1]) / 2
        pose_points_Lap = torch.from_numpy(pose_points_Lap.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        identity_faces=identity_faces
        pose_faces=pose_faces

        # print(pose_faces)
        # print(identity_faces)

        return pose_points, pose_faces, identity_points, identity_faces,pose_points_Lap


    def __len__(self):
        return self.length



class FAUST_DATA_LAP_with_GIH(data.Dataset):
    def __init__(self, train=True,  npoints=6890, shuffle_point = False, training_size = 400):
        self.train = train
        self.shuffle_point = shuffle_point
        self.npoints = npoints
        self.path='./dataset_3d/FAUST/'
        self.path_GIH_gt='./dataset_3d/FAUST_GIH_gt/'
        self.sample_size = len(os.listdir(self.path))
        self.length = training_size
        #self.test_label_path = './datasets/FAUST/FAUST_list/seen_pose_list.txt'
        #self.test_list = open(self.test_label_path,'r').read().splitlines()

    def __getitem__(self, index):
        identity_mesh_idx=np.random.randint(0,self.sample_size )
        pose_mesh_idx=np.random.randint(0,self.sample_size )

        identity_mesh_path = self.path+'tr_reg_'+str(identity_mesh_idx).zfill(3)+'.obj'
        pose_mesh_path = self.path+'tr_reg_'+str(pose_mesh_idx).zfill(3)+'.obj'

        pose_mesh_path = self.path+'tr_reg_'+str(pose_mesh_idx).zfill(3)+'.obj'

        gt_path = self.path_GIH_gt+'tr_reg_'+str(identity_mesh_idx).zfill(3)+'.pkl'
        with open(gt_path, "rb") as f:
            GIH_gt = pickle.load(f)
        GIH_gt = torch.from_numpy(GIH_gt.astype(np.float32))


        identity_mesh=trimesh_load_obj(identity_mesh_path)
        pose_mesh=trimesh_load_obj(pose_mesh_path)

        identity_points=identity_mesh.vertices
        identity_faces=identity_mesh.faces

        pose_points = pose_mesh.vertices
        pose_faces=pose_mesh.faces

        pose_points_Lap=pose_mesh.Lap_vertices
        pose_faces_Lap=pose_mesh.Lap_faces

        # print(identity_mesh)
        # print('hi')

        identity_points=identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points=torch.from_numpy(identity_points.astype(np.float32))

        pose_points=pose_points-(pose_mesh.bbox[0]+pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))

        pose_points_Lap=pose_points_Lap-(pose_mesh.Lap_bbox[0]+pose_mesh.Lap_bbox[1]) / 2
        pose_points_Lap = torch.from_numpy(pose_points_Lap.astype(np.float32))

        #if self.train:
        #    a = torch.FloatTensor(3)
        #    pose_points = pose_points + (a.uniform_(-1,1) * 0.03).unsqueeze(0).expand(-1, 3)

        #identity_faces=identity_faces
        #pose_faces=pose_faces

        # print(pose_faces)
        # print(identity_faces)

        return pose_points, pose_faces, identity_points, identity_faces, GIH_gt,pose_points_Lap


    def __len__(self):
        return self.length
