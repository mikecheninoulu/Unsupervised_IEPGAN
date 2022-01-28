from scipy import sparse
#import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.sparse.linalg import eigsh
from matplotlib import tri as mtri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from plyfile import PlyData, PlyElement
import time
import torch

def totuple(a):
    return [ tuple(i) for i in a]

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

def load_mesh(path, torch_tensors=False):
    VERT = np.loadtxt(path+'/mesh.vert')
    TRIV = np.loadtxt(path+'/mesh.triv',dtype='int32')-1
    if torch_tensors:
        VERT = torch.from_numpy(VERT)
        TRIV = torch.from_numpy(TRIV)
    
    return VERT, TRIV

def totuple(a):
    return [ tuple(i) for i in a]
    
def save_ply(V,T,filename):
    if(V.shape[1]==2):
        Vt = np.zeros((V.shape[0],3))
        Vt[:,0:2] = V
        V = Vt
        
    vertex = np.array(totuple(V),dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    face = np.array([ tuple([i]) for i in T],dtype=[('vertex_indices', 'i4', (3,))])
    el1 = PlyElement.describe(vertex, 'vertex')
    el2 = PlyElement.describe(face, 'face')
    PlyData([el1,el2]).write(filename)
    
    
def ismember(T, pts):
    out = np.zeros(np.shape(T)[0])
    for r in range(np.shape(T)[0]):
        s=0
        for c in range(np.shape(T)[1]):
            if np.sum(T[r,c]==pts)>0: s=s+1;
        out[r] = s>0;
    return out

def calc_adj_matrix(VERT,TRIV):
    n = np.shape(VERT)[0]
    A = np.zeros((n,n))    
    A[TRIV[:,0],TRIV[:,1]] = 1
    A[TRIV[:,1],TRIV[:,2]] = 1
    A[TRIV[:,2],TRIV[:,0]] = 1
    return A



def pairwise_dists(x):
    D = torch.sum( (x[:,None,:]-x[:,:,None])**2,-1)
    return D


"""
Drawing functions
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def plot_colormap(verts,trivs,cols, colorscale=[[0, 'rgb(0,0,255)'], [0.5, 'rgb(255,255,255)'], [1, 'rgb(255,0,0)']]):
    "Draw multiple triangle meshes side by side"
    if type(verts) is not list:
        verts=[verts]
    if type(trivs) is not list:
        trivs=[trivs]
    if type(cols) is not list:
        cols=[cols]
    
    
    nshapes = min([len(verts),len(cols),len(trivs)])
    
    fig = make_subplots(rows=1, cols=nshapes, specs=[[{'type': 'surface'} for i in range(nshapes)]])
    for i, [vert,triv,col] in enumerate(zip(verts,trivs,cols)):
        if col is not None:
            mesh = go.Mesh3d(x=vert[:,0], z=vert[:,1], y=vert[:,2],
                            i=triv[:,0], j=triv[:,1], k=triv[:,2],
                            intensity=col,
                            colorscale= colorscale,
                            color='lightpink', opacity=1)
        else:
            mesh = go.Mesh3d(x=vert[:,0], z=vert[:,1], y=vert[:,2],
                            i=triv[:,0], j=triv[:,1], k=triv[:,2])
            
        fig.add_trace(mesh,row=1,col=i+1)        
        fig.get_subplot(1,i+1).aspectmode="data"
      
        camera = dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=0, y=4, z=-1)
            )
        fig.get_subplot(1,i+1).camera=camera
      
    
#     fig = go.Figure(data=[mesh], layout=layout)
    fig.update_layout(
#       autosize=True,
      margin=dict(l=10, r=10, t=10, b=10),
      paper_bgcolor="LightSteelBlue")
    fig.show()
    return fig



#!/usr/bin/python

from numpy import *
from math import sqrt

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    A = np.matrix(A)
    B = np.matrix(B)
    num_rows, num_cols = A.shape;

    if num_rows != 3:
        raise Exception("matrix A is not 3xN, it is {}x{}".format(num_rows, num_cols))

    [num_rows, num_cols] = B.shape;
    if num_rows != 3:
        raise Exception("matrix B is not 3xN, it is {}x{}".format(num_rows, num_cols))

    # find mean column wise
    centroid_A = mean(A, axis=1)
    centroid_B = mean(B, axis=1)

    # ensure centroids are 3x1 (necessary when A or B are 
    # numpy arrays instead of numpy matrices)
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - tile(centroid_A, (1, num_cols))
    Bm = B - tile(centroid_B, (1, num_cols))

    H = Am * transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = linalg.svd(H)
    R = Vt.T * U.T

    # special reflection case
    if linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...\n");
        Vt[2,:] *= -1
        R = Vt.T * U.T

    t = -R*centroid_A + centroid_B

    return R, t




    
def decode(vae, device, lsp):
    lsp = torch.from_numpy(np.asarray(lsp,'float32')).to(device)
    xx= vae.dec(lsp)
    return xx.data#.cpu().numpy()

def calc_allsp(vae, device, dataset):
    alllsp=[]
    allx=[]

    
    vae.eval()
    
    for i in range(len(dataset)):
        data = dataset[i]
        allx.append(data.pos.data.cpu().numpy())
        
        lsp = vae.enc(data.oripos.to(device)[None,...])
        alllsp.append(lsp[...,:lsp.shape[-1]//2].data.cpu().numpy()[0])
#         alllsp.append(enc(x.cuda(),adj_input[:2,:],batch=adj_input[2,:])[...,:opt.LATENT_SPACE].data.cpu().numpy()[0])

    alllsp=np.asarray(alllsp)
    allx=np.asarray(allx)
    return alllsp, allx

