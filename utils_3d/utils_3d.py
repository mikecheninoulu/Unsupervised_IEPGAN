import numpy as np
import torch
import torch.nn as nn

def init_regul(source_vertices, source_faces):
    sommet_A_source = source_vertices[source_faces[:, 0]]
    sommet_B_source = source_vertices[source_faces[:, 1]]
    sommet_C_source = source_vertices[source_faces[:, 2]]
    target = []
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_B_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_B_source - sommet_C_source) ** 2, axis=1)))
    target.append(np.sqrt( np.sum((sommet_A_source - sommet_C_source) ** 2, axis=1)))
    # print(len(target))
    return target

def get_target(vertice, face, size):
    target = init_regul(vertice,face)
    target = np.array(target)
    target = torch.from_numpy(target).float().cuda()
    #target = target+0.0001
    target = target.unsqueeze(1).expand(3,size,-1)
    return target

def compute_score(points, faces, target):
    score = 0
    #print(points.shape)
    #print(faces.shape)
    #print(target.shape)
    sommet_A = points[:,faces[:, 0]]
    sommet_B = points[:,faces[:, 1]]
    sommet_C = points[:,faces[:, 2]]

    score = torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_B) ** 2, dim=2)) / target[0] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_B - sommet_C) ** 2, dim=2)) / target[1] -1)
    score = score + torch.abs(torch.sqrt(torch.sum((sommet_A - sommet_C) ** 2, dim=2)) / target[2] -1)
    #print(score)

    return torch.mean(score)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)



def central_distance_mean_score(points, gt_points, faces):
    score = 0

    for point_index in range(len(points)):
        # print(point_index)
        connected_trianlges = np.where(faces == point_index)[0]
        # print(connected_trianlges.shape)
        # print(connected_trianlges)
        connected_points_index = np.delete(np.unique(faces[connected_trianlges,:]), point_index)
        # print(connected_points_index)
        connected_points= points[:,connected_points_index]
        gt_connected_points= gt_points[:,connected_points_index]

        current_point_array = points[:,point_index].repeat(connected_points.shape[1], 1)
        gt_current_point_array = gt_points[:,point_index].repeat(connected_points.shape[1], 1)
        # print(current_point_array)

        distance = connected_points - current_point_array
        gt_distance = gt_connected_points - gt_current_point_array
        loss = nn.MSELoss()
        #print(distance.shape)
        #print(gt_distance.shape)
        score += loss(distance, gt_distance)

    return torch.mean(score)


def central_distance_mean_score_adaptive(points, gt_points, faces):
    score = 0

    adaptive_score= np.zeros(len(points))

    for point_index in range(len(points)):
        # print(point_index)
        connected_trianlges = np.where(faces == point_index)[0]
        # print(connected_trianlges.shape)
        # print(connected_trianlges)
        connected_points_index = np.delete(np.unique(faces[connected_trianlges,:]), point_index)
        # print(connected_points_index)
        connected_points= points[:,connected_points_index]
        gt_connected_points= gt_points[:,connected_points_index]

        current_point_array = points[:,point_index].repeat(connected_points.shape[1], 1)
        gt_current_point_array = gt_points[:,point_index].repeat(connected_points.shape[1], 1)
        # print(current_point_array)

        distance = connected_points - current_point_array
        gt_distance = gt_connected_points - gt_current_point_array
        loss = nn.MSELoss()
        #print(distance.shape)
        #print(gt_distance.shape)
        score += loss(distance, gt_distance)
        adaptive_score[point_index] = loss(distance, gt_distance)

    return torch.mean(score),adaptive_score


def central_distance_gradient_score(points, gt_points, faces):
    score = 0

    for point_index in range(len(points)):
        # print(point_index)
        connected_trianlges = np.where(faces == point_index)[0]
        # print(connected_trianlges.shape)
        # print(connected_trianlges)
        connected_points_index = np.delete(np.unique(faces[connected_trianlges,:]), point_index)
        connected_points= points[:,connected_points_index]
        gt_connected_points= gt_points[:,connected_points_index]
        # print(connected_points)
        # print(connected_points.shape)connected_points
        current_point_array = points[:,point_index].repeat(connected_points.shape[1], 1)
        gt_current_point_array = gt_points[:,point_index].repeat(connected_points.shape[1], 1)
        # print(current_point_array)

        score += torch.mean( torch.sqrt(torch.sum((connected_points - current_point_array) ** 2, dim=2)))

    return torch.mean(score)
