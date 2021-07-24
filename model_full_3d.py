import math

import torch
from torch import nn
from torch.nn import functional as F

# from stylegan2.model import StyledConv, Blur, EqualLinear, EqualConv2d, ScaledLeakyReLU
# from stylegan2.op import FusedLeakyReLU

class Encoder_3d(nn.Module):
    def __init__(self, num_points = 6890):
        super(Encoder_3d, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.conv_pose1 = torch.nn.Conv1d(512, 512, 1)
        self.conv_pose2 = torch.nn.Conv1d(512, 32, 1)
        self.conv_shape1 = torch.nn.Conv1d(512, 1024, 1)
        self.conv_shape2 = torch.nn.Conv1d(1024, 2048, 1)

        self.norm1 = torch.nn.InstanceNorm1d(64)
        self.norm2 = torch.nn.InstanceNorm1d(128)
        self.norm3 = torch.nn.InstanceNorm1d(512)
        self.norm4 = torch.nn.InstanceNorm1d(1024)
        self.norm5 = torch.nn.InstanceNorm1d(2048)
        self.norm6 = torch.nn.InstanceNorm1d(32)

        self.adaptivemaxpool = torch.nn.AdaptiveMaxPool2d((3, 6890))

    def forward(self, x):
        # print(x.shape)

        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))

        #pose
        x_pose = F.relu(self.norm3(self.conv_pose1(x)))
        x_pose = F.relu(self.norm6(self.conv_pose2(x_pose)))

        #shape
        x_shape = F.relu(self.norm4(self.conv_shape1(x)))
        x_shape = F.relu(self.norm5(self.conv_shape2(x_shape)))
        x_shape = self.adaptivemaxpool(x_shape)

        return x_pose,x_shape
#
# class Swap_Encoder(nn.Module):
#     def __init__(self, num_points = 6890):
#         super(Swap_Encoder, self).__init__()
#
#         self.conv1 = torch.nn.Conv1d(3, 64, 1)
#         self.conv2 = torch.nn.Conv1d(64, 128, 1)
#         self.conv3 = torch.nn.Conv1d(128, 1024, 1)
#
#         self.norm1 = torch.nn.InstanceNorm1d(64)
#         self.norm2 = torch.nn.InstanceNorm1d(128)
#         self.norm3 = torch.nn.InstanceNorm1d(1024)
#
#     def forward(self, x):
#
#         x = F.relu(self.norm1(self.conv1(x)))
#         x = F.relu(self.norm2(self.conv2(x)))
#         x = F.relu(self.norm3(self.conv3(x)))
#
#         return x

class SPAdaIN(nn.Module):
    def __init__(self,norm,input_nc,planes):
        super(SPAdaIN,self).__init__()
        self.conv_weight = nn.Conv1d(input_nc, planes, 1)
        self.conv_bias = nn.Conv1d(input_nc, planes, 1)
        self.norm = norm(planes)

    def forward(self,x,addition):

        x = self.norm(x)
        weight = self.conv_weight(addition)
        bias = self.conv_bias(addition)
        out =  weight * x + bias

        return out

class SPAdaINResBlock(nn.Module):
    def __init__(self,input_nc,planes,norm=nn.InstanceNorm1d,conv_kernel_size=1,padding=0):
        super(SPAdaINResBlock,self).__init__()
        self.spadain1 = SPAdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(planes, planes, kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spadain2 = SPAdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.conv2 = nn.Conv1d(planes,planes,kernel_size=conv_kernel_size, stride=1, padding=padding)
        self.spadain_res = SPAdaIN(norm=norm,input_nc=input_nc,planes=planes)
        self.conv_res=nn.Conv1d(planes,planes,kernel_size=conv_kernel_size, stride=1, padding=padding)

    def forward(self,x,addition):

        out = self.spadain1(x,addition)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.spadain2(out,addition)
        out = self.relu(out)
        out = self.conv2(out)

        residual = x
        residual = self.spadain_res(residual,addition)
        residual = self.relu(residual)
        residual = self.conv_res(residual)

        out = out + residual

        return  out


class Generator_3d(nn.Module):
    def __init__(self, bottleneck_size = 32):
        self.bottleneck_size = bottleneck_size
        super(Generator_3d, self).__init__()

        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.spadain_block1 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size)
        self.spadain_block2 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//2)
        self.spadain_block3 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//4)

        self.norm1 = torch.nn.InstanceNorm1d(self.bottleneck_size)
        self.norm2 = torch.nn.InstanceNorm1d(self.bottleneck_size//2)
        self.norm3 = torch.nn.InstanceNorm1d(self.bottleneck_size//4)
        self.th = nn.Tanh()


    def forward(self, x, addition):

        x = self.conv1(x)
        x = self.spadain_block1(x,addition)
        x = self.conv2(x)
        x = self.spadain_block2(x,addition)
        x = self.conv3(x)
        x = self.spadain_block3(x,addition)
        x = 2*self.th(self.conv4(x))

        return x



class NPT(nn.Module):
    def __init__(self, num_points = 6890, bottleneck_size = 1024):
        super(NPT, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.encoder = PoseFeature(num_points = num_points)
        self.decoder = Decoder(bottleneck_size = self.bottleneck_size+3)

    def forward(self, x1, x2):

        x1_f = self.encoder(x1)
        y = torch.cat((x1_f, x2), 1)
        out =self.decoder(y,x2)

        return x1_f, out.transpose(2,1)



class Discriminator_3d(nn.Module):
    def __init__(self, bottleneck_size = 1024):
        self.bottleneck_size = bottleneck_size
        super(Discriminator_3d, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, self.bottleneck_size//4, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size//4, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size//2, self.bottleneck_size, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        # self.conv6 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.spadain_block1 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//4)
        self.spadain_block2 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size//2)
        self.spadain_block3 = SPAdaINResBlock(input_nc=3 ,planes=self.bottleneck_size)

        self.norm1 = torch.nn.InstanceNorm1d(self.bottleneck_size//4)
        self.norm2 = torch.nn.InstanceNorm1d(self.bottleneck_size//2)
        self.norm3 = torch.nn.InstanceNorm1d(self.bottleneck_size)

        self.avgpool = nn.AdaptiveAvgPool2d((4,512))
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(2048, 1024)
        self.dense2 = nn.Linear(1024, 1)
        # self.bilinear1= torch.nn.Bilinear(self.bottleneck_size, self.bottleneck_size, self.bottleneck_size//4)
        # self.bilinear2= torch.nn.Bilinear(6890, 6890, 6890//10)
        self.th = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x_mesh):

        x = self.conv1(x_mesh)
        # x = self.relu(x)
        x = self.spadain_block1(x,x_mesh)
        x = self.conv2(x)
        # x = self.relu(x)
        x = self.spadain_block2(x,x_mesh)
        x = self.conv3(x)
        # x = self.relu(x)
        x = self.spadain_block3(x,x_mesh)
        x = self.conv4(x)
        x = self.th(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        return x




class CooccurDiscriminator(nn.Module):
    def __init__(self):
        super(CooccurDiscriminator, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 256, 1)

        self.norm1 = torch.nn.InstanceNorm1d(64)
        self.norm2 = torch.nn.InstanceNorm1d(128)
        self.norm3 = torch.nn.InstanceNorm1d(256)

        self.dense1 = torch.nn.Linear(256, 1024)
        self.dense2 = torch.nn.Linear(1024, 1)

        #self.num_points = num_points
        self.adappooling = torch.nn.AdaptiveMaxPool2d((16,16))

    def forward(self, input, reference=None, ref_batch=None, ref_input=None):

        out_input = F.relu(self.norm1(self.conv1(input)))
        out_input = F.relu(self.norm2(self.conv2(out_input)))
        out_input = F.relu(self.norm3(self.conv3(out_input)))
        out_input = F.relu(self.norm3(self.conv4(out_input)))

        if ref_input is None:
            ref_input = F.relu(self.norm1(self.conv1(reference)))
            ref_input = F.relu(self.norm2(self.conv2(ref_input)))
            ref_input = F.relu(self.norm3(self.conv3(ref_input)))
            ref_input = F.relu(self.norm3(self.conv4(ref_input)))

            _, height, width = ref_input.shape
            # print(ref_input.shape)
            # print(ref_batch)
            ref_input = ref_input.view(-1, ref_batch, height, width)
            # print(ref_input.shape)
            ref_input = ref_input.mean(1)
            # print(ref_input.shape)

        out = torch.cat((out_input, ref_input), 1)
        out = self.adappooling(out)


        out = torch.flatten(out, 1)

        # print(out.shape)

        out = F.relu(self.dense1(out))
        out = self.dense2(out)

        return out,ref_input
