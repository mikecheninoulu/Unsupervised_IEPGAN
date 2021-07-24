import argparse
import math
import random
import os
#os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import open3d as o3d
# from utils_3d.weak_perspective_pyrender_renderer import Renderer
import cv2
import utils_3d.utils_3d as utils_3d
import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from utils_3d.objLoader_trimesh import trimesh_load_obj
from utils_3d.utils_distance import *
import copy
try:
    import wandb

except ImportError:
    wandb = None

from model_full_3d import Encoder_3d, Generator_3d,Discriminator_3d, CooccurDiscriminator
from utils_3d.dataset_3d import FAUST_DATA_with_GIH,FAUST_DATA,SMPL_DATA
from utils_3d.distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def vae_loss(logSigmaSquare, sigmaSquare, muSquare):
    return -0.5 * (torch.ones_like(muSquare) + logSigmaSquare - sigmaSquare  - muSquare)


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    (grad_real,) = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None

def travel_connected_points(connected_points,mesh_faces):

    connected_trianlges_index_total = []
    for point_index in connected_points:
        connected_trianlges_index = np.where(mesh_faces == point_index)[0]
        connected_trianlges_index_total = np.concatenate((connected_trianlges_index_total, connected_trianlges_index))
        connected_trianlges_index_total = connected_trianlges_index_total.astype(int)
    connected_points = np.unique(mesh_faces[connected_trianlges_index_total,:])
    return connected_points, connected_trianlges_index_total

def patchify_mesh(mesh, limb_data_total,sampling_number,Ref,crop_n=2):
    patches = []
    # print(mesh.shape)#(2, 3, 6890)

    if Ref == True:
        limb_data_total = limb_data_total
    else:
        limb_data_total = random.sample(limb_data_total,crop_n)
    # print(mesh.shape)
    for limb_data in limb_data_total:
        # print(limb_data['faces'].shape)
        # print(limb_data['points'].shape)

        cropped = mesh[:,:,limb_data['points']].unsqueeze(1)
        # print(cropped.shape)

        cropped = F.interpolate(
            cropped, size=(3,sampling_number), mode="bilinear", align_corners=False
        ).squeeze(1)

        # print(cropped.shape)

        patches.append(cropped)

    patches = torch.stack(patches, 1).view(-1, 3, sampling_number)

    print(patches.shape)#([4, 3, 1000])

    return patches

def spectralify_mesh(mesh,faces,sampling_number,sample_n=2, min_ite=1, max_ite=20):

    patches = []
    #print(sampling_number)
    for batch_i in range(mesh.shape[0]):
        newmesh = o3d.geometry.TriangleMesh()
        newmesh.vertices = o3d.utility.Vector3dVector(np.float64(mesh[batch_i]).transpose(1,0))
        newmesh.triangles = o3d.utility.Vector3iVector(faces[batch_i])
        for sample_index in range(sample_n):
            n_iter = np.random.randint((max_ite-min_ite)) + min_ite
            #print(n_iter)
            pyramid_mesh = copy.deepcopy(newmesh)
            pyramid_mesh = pyramid_mesh.filter_smooth_laplacian(number_of_iterations=n_iter)
            pyramid_mesh.compute_vertex_normals()
            #print(len(pyramid_mesh.vertices))
            # pyramid_mesh = pyramid_mesh.simplify_quadric_decimation(target_number_of_triangles=sampling_number)

            pyramid_mesh = pyramid_mesh.sample_points_poisson_disk(number_of_points= sampling_number)
            vertices = np.array(pyramid_mesh.points)
            # triangles = np.array(pyramid_mesh.triangles)
            #print(vertices.shape)
            # print(triangles.shape)
            patches.append(vertices)
    patches = torch.FloatTensor(patches).to('cuda')
    #print(patches.shape)
    patches = patches.view(-1, 3, sampling_number)

    #print(patches.shape)
    return patches


def train(
    args,
    loader_rec,
    loader_geo,
    encoder,
    generator,
    discriminator,
    cooccur,
    g_optim,
    d_optim,
    e_ema,
    g_ema,
    device,
    rec_epoch,
    geo_epoch,
    stage_setting,
    experi_path,
    server,
    geoloss_weight,
    sampling_number,
    n_crop,
    limb_n,
    limb_sampling,
):
    loader_rec = sample_data(loader_rec)
    loader_geo = sample_data(loader_geo)

    limb_path = 'Limb_data/'
    limb_list = os.listdir(limb_path)

    limb_data_total = []
    for limb_file in limb_list:
        limb_data = np.load(limb_path + limb_file)
        limb_data_total.append(limb_data)

    pbar = range(args.iter+1)

    edge_loss_setting = stage_setting

    '''make experi folder'''
    if server == 'local':
        experiment_path = 'experis/'+ experi_path +'/'
    elif server == 'csc':
         experiment_path = '/scratch/project_2003217/swap3d/checkpoint/'+ experi_path +'/'
    else:
        print('error')

    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)


    # wp_renderer = Renderer(resolution=(128, 128))

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    loss_dict = {}

    if args.distributed:
        e_module = encoder.module
        g_module = generator.module
        d_module = discriminator.module
        c_module = cooccur.module

    else:
        e_module = encoder
        g_module = generator
        d_module = discriminator
        c_module = cooccur

    accum = 0.5 ** (32 / (10 * 1000))

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        if i > geo_epoch:
            pose_points_raw, pose_faces_raw, identity_points_raw, _, GIH_gt = next(loader_geo)#,Dg_t1, Dg_t2 = next(loader)
            shape_GIH_gt = GIH_gt.to(device)
            gt_faces = pose_faces_raw.long().to(device)
        else:
            pose_points_raw, pose_faces_raw, identity_points_raw, _,gt_points_raw = next(loader_rec)#,Dg_t1, Dg_t2 = next(loader)

        pose_points=pose_points_raw.transpose(2,1)
        identity_points=identity_points_raw.transpose(2,1)
        gt_points=gt_points_raw.transpose(2,1)

        real_mesh1 = pose_points.to(device)
        real_mesh2 = identity_points.to(device)
        gt_mesh = gt_points.to(device)
        real_meshes = torch.cat((real_mesh1, real_mesh2), 0)

        requires_grad(encoder, False)
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        requires_grad(cooccur, True)

        # print(identity_points.shape)
        # print(pose_points.shape)
        #real_img1, real_img2 = real_img.chunk(2, dim=0)
        # print(real_img1.shape)
        # print(real_img2.shape)#32*32
        structure1,shape1 = encoder(real_mesh1)
        structure2,shape2 = encoder(real_mesh2)

        fake_mesh1 = generator(structure1, shape1)
        fake_mesh2 = generator(structure1, shape2)

        fake_pred = discriminator(torch.cat((fake_mesh1, fake_mesh2), 0))
        real_pred = discriminator(real_meshes)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        fake_patch = spectralify_mesh(fake_mesh2.cpu().numpy(), pose_faces_raw.cpu().numpy(),sampling_number,n_crop)
        real_patch = spectralify_mesh(real_mesh1.cpu().numpy(), pose_faces_raw.cpu().numpy(),sampling_number,n_crop)
        ref_patch = spectralify_mesh(real_mesh1.cpu().numpy(), pose_faces_raw.cpu().numpy(),sampling_number,args.ref_crop * n_crop)

        # print(cooccur)
        # print(fake_patch.shape)#torch.Size([4, 3, 60])
        # print(ref_patch.shape)#torch.Size([12, 3, 60])


        fake_patch_pred, ref_input = cooccur(fake_patch, ref_patch, ref_batch = args.ref_crop)
        real_patch_pred, _ = cooccur(real_patch, ref_input=ref_input)
        cooccur_loss = d_logistic_loss(real_patch_pred, fake_patch_pred)#*0.0005

        # print(cooccur_loss)

        loss_dict["d"] = d_loss
        loss_dict["cooccur"] = cooccur_loss
        loss_dict["real_score"] = real_pred.mean()
        fake_pred1, fake_pred2 = fake_pred.chunk(2, dim=0)
        loss_dict["fake_score"] = fake_pred1.mean()
        loss_dict["hybrid_score"] = fake_pred2.mean()

        d_optim.zero_grad()
        (d_loss + cooccur_loss).backward()
        # d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_meshes.requires_grad = True
            real_pred = discriminator(real_meshes)
            r1_loss = d_r1_loss(real_pred, real_meshes)

            real_patch.requires_grad = True
            real_patch_pred, _ = cooccur(real_patch, ref_patch, ref_batch=args.ref_crop)
            cooccur_r1_loss = d_r1_loss(real_patch_pred, real_patch)
            d_optim.zero_grad()

            r1_loss_sum = args.r1 / 2 * r1_loss * args.d_reg_every
            r1_loss_sum += args.cooccur_r1 / 2 * cooccur_r1_loss * args.d_reg_every
            r1_loss_sum.backward()

            d_optim.step()

        loss_dict["r1"] = r1_loss
        loss_dict["cooccur_r1"] = cooccur_r1_loss

        requires_grad(encoder, True)
        requires_grad(generator, True)
        requires_grad(discriminator, False)
        requires_grad(cooccur, False)

        real_meshes.requires_grad = False

        # structure1 = encoder(real_mesh1)
        # structure2 = encoder(real_mesh2)

        structure1,shape1 = encoder(real_mesh1)
        structure2,shape2 = encoder(real_mesh2)

        fake_mesh1 = generator(structure1, shape1)
        fake_mesh2 = generator(structure1, shape2)

        #vae_loss_final = vae_loss1.mean() + vae_loss2.mean()
        #print (vae_loss_final)
        if i < geo_epoch:
            recon_loss = F.l1_loss(fake_mesh2, gt_mesh)
        else:
            recon_loss = F.l1_loss(fake_mesh1, real_mesh1)
        #print (recon_loss)


        if edge_loss_setting  == 'rec_swap':
            stage1_edge_mesh = fake_mesh1
            stage1_edge_mesh_gt  = real_mesh1

            stage2_edge_mesh = fake_mesh2
            stage2_edge_mesh_gt  = real_mesh2

        elif edge_loss_setting == 'rec_rec':
            stage1_edge_mesh = fake_mesh1
            stage1_edge_mesh_gt  = real_mesh1

            stage2_edge_mesh = fake_mesh1
            stage2_edge_mesh_gt  = real_mesh1
        elif edge_loss_setting == 'swap_swap':
            stage1_edge_mesh = fake_mesh2
            stage1_edge_mesh_gt  = real_mesh2

            stage2_edge_mesh = fake_mesh2
            stage2_edge_mesh_gt  = real_mesh2
        else:
            print('error edge_loss_setting')
        # print('edg_loss')
        # print(edg_loss)

        if i<=rec_epoch:
            recon_loss=recon_loss# + vae_loss_final
        if i>rec_epoch and i<= geo_epoch:
            '''CDC loss'''
            central_distance_loss= 0
            for i_face in range(len(fake_mesh1)):
                f=pose_faces_raw[i_face].cpu().numpy()
                v=real_mesh1[i_face].unsqueeze(0)
                # print(v.shape)#(1,6890, 3)
                central_distance_loss += utils_3d.central_distance_mean_score(fake_mesh1[i_face].transpose(0,1).unsqueeze(0),v.transpose(1,2),f)
            central_distance_loss=central_distance_loss/len(fake_mesh1)

            '''edge loss'''
            edg_loss= 0
            for i_face in range(len(pose_faces_raw)):
                f=pose_faces_raw[i_face].cpu().numpy()
                v=stage1_edge_mesh_gt[i_face].transpose(0,1).cpu().numpy()
                edg_loss=edg_loss+utils_3d.compute_score(stage1_edge_mesh[i_face].transpose(0,1).unsqueeze(0),f,utils_3d.get_target(v,f,1))
            edg_loss=edg_loss/len(pose_faces_raw)
            recon_loss=recon_loss+0.0005*central_distance_loss+0.0005*edg_loss# + vae_loss_final#

        if i > geo_epoch:

            '''CGC loss'''
            central_distance_loss= 0
            adaptive_map = np.zeros(fake_mesh1.shape[2])
            #print(adaptive_map.shape)
            for i_face in range(len(fake_mesh1)):
                f=pose_faces_raw[i_face].cpu().numpy()
                v=real_mesh1[i_face].unsqueeze(0)
                # print(v.shape)#(1,6890, 3)
                CDC_loss, adaptive_map_local = utils_3d.central_distance_mean_score_adaptive(fake_mesh1[i_face].transpose(0,1).unsqueeze(0),v.transpose(1,2),f)
                central_distance_loss += CDC_loss
                adaptive_map = adaptive_map + adaptive_map_local
            # print(adaptive_map.shape)
            central_distance_loss=central_distance_loss/len(fake_mesh1)

            '''edge loss'''
            edg_loss= 0
            for i_face in range(len(pose_faces_raw)):
                f=pose_faces_raw[i_face].cpu().numpy()
                v=stage2_edge_mesh_gt[i_face].transpose(0,1).cpu().numpy()
                edg_loss=edg_loss+utils_3d.compute_score(stage2_edge_mesh[i_face].transpose(0,1).unsqueeze(0),f,utils_3d.get_target(v,f,1))
            edg_loss=edg_loss/len(pose_faces_raw)

            '''GIH loss'''
            '''get regional meshes'''
            #print('hi')
            if args.sampling_pattern == 'random':
                limb_core_points = np.random.randint(6890, size=(limb_n, 1))
            elif args.sampling_pattern == 'adaptive':
                limb_core_points = (-adaptive_map).argsort()[:limb_n]
                limb_core_points = limb_core_points.reshape((limb_n, 1))
                # print(limb_core_points)
            else:
                print('pattern error')

            template_mesh_faces=pose_faces_raw[0]

            limb_data_total = []
            for limb_index in limb_core_points:
                connected_points = limb_index
                while len(connected_points)<limb_sampling:
                    connected_points,connected_trianlges_index_total = travel_connected_points(connected_points,template_mesh_faces)

                # print(connected_points.shape)
                new_point_index = 0
                old_limb_vertices = np.zeros(len(connected_points), dtype = np.long)
                new_limb_faces = template_mesh_faces[connected_trianlges_index_total]
                for old_point in connected_points:
                    old_limb_vertices[new_point_index] = old_point
                    new_limb_faces = np.where(new_limb_faces==old_point, new_point_index, new_limb_faces)
                    new_point_index +=1
                limb_data_total.append([new_limb_faces,old_limb_vertices])

            # print(len(limb_data_total))
            '''calculate GIH regional meshes'''
            for i_face in range(len(pose_faces_raw)):
                for limb_data in limb_data_total:
                    limb_vertices  = limb_data[1]
                    limb_faces = torch.from_numpy(limb_data[0]).long().to(device)
                    Dg_r, grad, div, W, S, C = distance_GIH(fake_mesh2[i_face][:,limb_vertices].transpose(0,1).unsqueeze(0), limb_faces.unsqueeze(0))
                    geoloss = torch.mean( ((shape_GIH_gt[i_face][limb_vertices,limb_vertices]-Dg_r.float()))**2)
                    del limb_faces
                    del limb_vertices
            geoloss=geoloss/(len(pose_faces_raw) *len(limb_data_total))
            recon_loss=recon_loss+0.0005*central_distance_loss+0.0005*edg_loss+geoloss_weight*geoloss #+ vae_loss_final#

        fake_pred = discriminator(torch.cat((fake_mesh1, fake_mesh2), 0))
        g_loss = g_nonsaturating_loss(fake_pred)

        fake_patch = spectralify_mesh(fake_mesh2.cpu().detach().numpy(), pose_faces_raw.cpu().numpy(),sampling_number,n_crop)
        ref_patch = spectralify_mesh(real_mesh1.cpu().numpy(), pose_faces_raw.cpu().numpy(),sampling_number,args.ref_crop * n_crop)
        fake_patch_pred, _ = cooccur(fake_patch, ref_patch, ref_batch=args.ref_crop)
        g_cooccur_loss = g_nonsaturating_loss(fake_patch_pred)

        loss_dict["recon"] = recon_loss
        loss_dict["g"] = g_loss
        loss_dict["g_cooccur"] = g_cooccur_loss

        g_optim.zero_grad()
        (recon_loss + g_loss + g_cooccur_loss).backward()
        # (recon_loss + g_loss).backward()
        g_optim.step()

        accumulate(e_ema, e_module, accum)
        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        cooccur_val = loss_reduced["cooccur"].mean().item()
        recon_val = loss_reduced["recon"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        g_cooccur_val = loss_reduced["g_cooccur"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        cooccur_r1_val = loss_reduced["cooccur_r1"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        hybrid_score_val = loss_reduced["hybrid_score"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; c: {cooccur_val:.4f} g: {g_loss_val:.4f}; "
                    f"g_cooccur: {g_cooccur_val:.4f}; recon: {recon_val:.4f}; r1: {r1_val:.4f}; "
                    f"r1_cooccur: {cooccur_r1_val:.4f}"
                )
            )

            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; "
                    f"recon: {recon_val:.4f}; r1: {r1_val:.4f}; "
                    f"r1_cooccur: {cooccur_r1_val:.4f}"
                )
            )

            if wandb and args.wandb and i % 10 == 0:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Cooccur": cooccur_val,
                        "Recon": recon_val,
                        "Generator Cooccur": g_cooccur_val,
                        "R1": r1_val,
                        "Cooccur R1": cooccur_r1_val,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Hybrid Score": hybrid_score_val,
                    },
                    step=i,
                )

        if i % 200 == 0:
            with torch.no_grad():
                e_ema.eval()
                g_ema.eval()

                structure1, shape1 = e_ema(real_mesh1)
                structure2, shape2 = e_ema(real_mesh2)

                fake_mesh1 = g_ema(structure1, shape1)
                fake_mesh2 = g_ema(structure1, shape2)

                sample = torch.cat((fake_mesh1, fake_mesh2), 0)
                # print(fake_mesh1[0].detach().cpu().numpy().squeeze().shape)
                # print(pose_faces[0].detach().cpu().numpy().shape)
                rend_img = wp_renderer.render(
                verts = fake_mesh2[0].detach().cpu().numpy().transpose(1,0),
                faces = pose_faces_raw[0].detach().cpu().numpy(),
                cam=np.array([0.8, 0., 0.2]),
                angle=-180,
                axis= [1, 0, 0])
                cv2.imwrite(f"./sample_3d_raw/{str(i).zfill(6)}_fake.png", rend_img)

                rend_img = wp_renderer.render(
                verts = real_mesh2[0].detach().cpu().numpy().transpose(1,0),
                faces = pose_faces_raw[0].detach().cpu().numpy(),
                cam=np.array([0.8, 0., 0.2]),
                angle=-180,
                axis= [1, 0, 0])

                cv2.imwrite(f"./sample_3d_raw/{str(i).zfill(6)}_shape.png", rend_img)

                rend_img = wp_renderer.render(
                verts = real_mesh1[0].detach().cpu().numpy().transpose(1,0),
                faces = pose_faces_raw[0].detach().cpu().numpy(),
                cam=np.array([0.8, 0., 0.2]),
                angle=-180,
                axis= [1, 0, 0])

                cv2.imwrite(f"./sample_3d_raw/{str(i).zfill(6)}_pose.png", rend_img)

                # utils.save_image(
                #     sample,
                #     f"sample/{str(i).zfill(6)}.png",
                #     nrow=int(sample.shape[0] ** 0.5),
                #     normalize=True,
                #     range=(-1, 1),
                # )
        #print(i)
        if i % 500 == 0:
            #print('please save')
            torch.save(
                {
                    "e": e_module.state_dict(),
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "cooccur": c_module.state_dict(),
                    "e_ema": e_ema.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "args": args,
                },experiment_path +str(i)+ '.pt')

                   # f"/scratch/project_2003217/swap3d/checkpointedgeloss/{str(i).zfill(6)}.pt",
                #)
                # f"/scratch/project_2003217/swap3d/checkpoint/edgeloss{str(i).zfill(6)}.pt",

if __name__ == "__main__":
    device = "cuda"
    #print('right')

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str, nargs="+")
    parser.add_argument("--iter", type=int, default=40000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--cooccur_r1", type=float, default=1)
    parser.add_argument("--ref_crop", type=int, default=2)
    parser.add_argument("--n_crop", type=int, default=4)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--channel", type=int, default=32)
    parser.add_argument("--channel_multiplier", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--rec_epoch", type=int, default=8000)
    parser.add_argument("--geo_epoch", type=int, default=15000)
    parser.add_argument("--edge_loss_setting", type=str, default='swap_swap')
    parser.add_argument("--experi_path", type=str, default='without_GIH_swap_swap')
    parser.add_argument("--server", type=str, default='local')
    parser.add_argument("--geoloss", type=float, default=1)
    parser.add_argument("--sampling_number", type=int, default=1000)
    parser.add_argument("--limb_n", type=int, default=4)
    parser.add_argument("--limb_sampling", type=int, default=200)
    parser.add_argument("--sampling_pattern", type=str, default='adaptive')
    args = parser.parse_args()

    # os.mkdirs()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0

    stage_setting = args.edge_loss_setting

    encoder = Encoder_3d().to(device)
    # print(Encoder_3d())
    generator = Generator_3d().to(device)

    discriminator = Discriminator_3d().to(device)
    cooccur = CooccurDiscriminator().to(device)

    e_ema = Encoder_3d().to(device)
    g_ema = Generator_3d().to(device)
    e_ema.eval()
    g_ema.eval()
    accumulate(e_ema, encoder, 0)
    accumulate(g_ema, generator, 0)

    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        list(encoder.parameters()) + list(generator.parameters()),
        lr=args.lr,
        betas=(0, 0.99),
    )
    d_optim = optim.Adam(list(discriminator.parameters()),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        encoder.load_state_dict(ckpt["e"])
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        cooccur.load_state_dict(ckpt["cooccur"])
        e_ema.load_state_dict(ckpt["e_ema"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])

    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )


    dataset_rec = SMPL_DATA()
    dataset_geo = FAUST_DATA_with_GIH()

    loader_rec = data.DataLoader(
        dataset_rec,
        batch_size=args.batch,
        sampler=data_sampler(dataset_rec, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    loader_geo = data.DataLoader(
        dataset_geo,
        batch_size=args.batch,
        sampler=data_sampler(dataset_geo, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="swapping 3d autoencoder")

    train(
        args,
        loader_rec,
        loader_geo,
        encoder,
        generator,
        discriminator,
        cooccur,
        g_optim,
        d_optim,
        e_ema,
        g_ema,
        device,
        args.rec_epoch,
        args.geo_epoch,
        stage_setting,
        args.experi_path,
        args.server,
        args.geoloss,
        args.sampling_number,
        args.n_crop,
        args.limb_n,
        args.limb_sampling,
    )
