"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

import torch
from torch import nn
from torch.nn import functional as F

from second.pytorch.models.voxel_encoder import get_paddings_indicator, register_vfe
from second.pytorch.models.middle import register_middle
from torchplus.nn import Empty
from torchplus.tools import change_default_args
import numpy as np 

class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = 'PFNLayer'
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if use_norm:      #这个是将(D,N,P)转化为（C,W,H）?
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)           #eps：ε，默认是1e-5，momentum： 计算running_mean和running_var的滑动平均系数。
            Linear = change_default_args(bias=False)(nn.Linear)    #输入和权重矩阵相乘
            BatchNorm1dradar = change_default_args(
                eps=1e-3, momentum=0.01)(nn.BatchNorm1d)  # eps：ε，默认是1e-5，momentum： 计算running_mean和running_var的滑动平均系数。
            Linearradar = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)


        self.linear = Linear(in_channels, self.units)    #in_channels=9  units=64
        self.norm = BatchNorm1d(self.units)
        self.linearradar = Linearradar(in_channels, self.units)  # in_channels=9  units=64
        self.normradar = BatchNorm1dradar(self.units)

    def forward(self, inputs,batch_size):
        if batch_size==3:
            number=8000
        else:
            number=2500
        if inputs.shape[0]>number:
            x = self.linear(inputs)        #input:[9,60,41924]   x:[64,60,41924]
            x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                               1).contiguous()   #permute进行维度换位 [41924,60,64]
            x = F.relu(x)

            x_max = torch.max(x, dim=1, keepdim=True)[0]   #[41924,1,64]，dim=1是缩小的尺寸， [0]是输出值，[1]是输出索引

            if self.last_vfe:
                return x_max
            else:
                x_repeat = x_max.repeat(1, inputs.shape[1], 1)    #沿着特定的维度重复这个张量
                x_concatenated = torch.cat([x, x_repeat], dim=2)
                return x_concatenated
        else:
            x = self.linearradar(inputs)  # input:[9,60,41924]   x:[64,60,41924]
            x = self.normradar(x.permute(0, 2, 1).contiguous()).permute(0, 2,
                                                                   1).contiguous()  # permute进行维度换位 [41924,60,64]
            x = F.relu(x)

            x_max = torch.max(x, dim=1, keepdim=True)[0]  # [41924,1,64]，dim=1是缩小的尺寸， [0]是输出值，[1]是输出索引

            if self.last_vfe:
                return x_max
            else:
                x_repeat = x_max.repeat(1, inputs.shape[1], 1)  # 沿着特定的维度重复这个张量
                x_concatenated = torch.cat([x, x_repeat], dim=2)
                return x_concatenated


@register_vfe
class PillarFeatureNetOld(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNetOld'
        assert len(num_filters) > 0
        num_input_features += 5
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNetOld layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device

        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = features[:, :, :2]
        f_center[:, :, 0] = f_center[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = f_center[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()

@register_vfe
class PillarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNetOld'
        assert len(num_filters) > 0
        num_input_features += 5 #x,y,z,r,xp,yp,zp,xc,yc
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance   #false
        # self.is_radar=is_radar
        # Create PillarFeatureNetOld layers
        num_filters = [num_input_features] + list(num_filters) #[9,64]  所以out是64
        pfn_layers = [] #PFNLayer((linear): DefaultArgLayer(in_features=9, out_features=64, bias=False)(norm): DefaultArgLayer(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True))
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i] #in_filters 9
            out_filters = num_filters[i + 1] #out_filter 64
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            # if is_radar = True:
            #     is_radar=True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, use_norm,last_layer=last_layer))   #last_layer is flase
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset需要柱子(体素)尺寸和x/y偏移量来计算柱子偏移量
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors,batch_size): #coors坐标
        device = features.device

        dtype = features.dtype         #float32
        # Find distance of x, y, and z from cluster center 求x, y, z到聚类中心的距离
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        aa=features[:, :, :3].sum(
            dim=1, keepdim=True)                             #每个柱子60个点相加，只加x,y,z
        bb=num_voxels.type_as(features).view(-1, 1, 1).shape
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center 求x, y, z到柱子中心的距离
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations 组合特征装饰
        if features.shape[2] == 4:
            features_ls = [features, f_cluster, f_center]        #加上到中心点距离和到底面中心点距离变成9维度
        if features.shape[2] == 6:
            features_ls = [features, f_cluster]
        if self._with_distance:            #false
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1) #feature shape
        a=features.shape    #torch.Size([50929, 60, 9])    #[3444,60,23]radar

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.特征装饰的计算不考虑柱子是否空。需要确保
        # 空柱子仍然设置为零
        voxel_count = features.shape[1]            #一个柱子60的点
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)   #根据voxel的点数设置mask
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask    #特征mask掩膜(60,9,41924)

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features,batch_size)
            # print(features.shape)           #[38093,1,64]

        return features.squeeze()



class PillarRadarFeatureNet(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        super().__init__()
        self.name = 'PillarFeatureNetOld'
        assert len(num_filters) > 0
        num_input_features += 5  # x,y,z,r,vx,vy,xc,yc,zc
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance  # false

        # Create PillarFeatureNetOld layers
        num_filters = [num_input_features] + list(num_filters)  # [9,64]  所以out是64
        pfn_layers = []  # PFNLayer((linear): DefaultArgLayer(in_features=9, out_features=64, bias=False)(norm): DefaultArgLayer(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True))
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]  # in_filters 9
            out_filters = num_filters[i + 1]  # out_filter 64
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, use_norm, last_layer=last_layer))  # last_layer is flase
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset需要柱子(体素)尺寸和x/y偏移量来计算柱子偏移量
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]
    def forward(self, features, num_voxels, coors): #coors坐标
        device = features.device

        dtype = features.dtype         #float32
        # Find distance of x, y, and z from cluster center 求x, y, z到聚类中心的距离
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        aa=features[:, :, :3].sum(
            dim=1, keepdim=True)                             #每个柱子60个点相加，只加x,y,z
        bb=num_voxels.type_as(features).view(-1, 1, 1).shape
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center 求x, y, z到柱子中心的距离
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations 组合特征装饰
        features_ls = [features, f_cluster,f_center]        #加上到中心点距离和到底面中心点距离变成9维度
        if self._with_distance:            #false
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1) #feature shape
        a=features.shape    #torch.Size([50929, 60, 9])    #[3444,60,23]radar

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.特征装饰的计算不考虑柱子是否空。需要确保
        # 空柱子仍然设置为零
        voxel_count = features.shape[1]            #一个柱子60的点
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)   #根据voxel的点数设置mask
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask    #特征mask掩膜(60,9,41924)

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)
            # print(features.shape)           #[38093,1,64]

        return features.squeeze()


@register_vfe
class PillarFeatureNetRadius(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNetRadius'
        assert len(num_filters) > 0
        num_input_features += 5
        num_input_features -= 1 # radius xy->r, z->z
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNetOld layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device

        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
        features_radius = torch.norm(features[:, :, :2], p=2, dim=2, keepdim=True)
        features_radius = torch.cat([features_radius, features[:, :, 2:]], dim=2)
        # Combine together feature decorations
        features_ls = [features_radius, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()

@register_vfe
class PillarFeatureNetRadiusHeight(nn.Module):
    def __init__(self,
                 num_input_features=4,
                 use_norm=True,
                 num_filters=(64, ),
                 with_distance=False,
                 voxel_size=(0.2, 0.2, 4),
                 pc_range=(0, -40, -3, 70.4, 40, 1)):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param use_norm: <bool>. Whether to include BatchNorm.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = 'PillarFeatureNetRadiusHeight'
        assert len(num_filters) > 0
        num_input_features += 6
        num_input_features -= 1 # radius xy->r, z->z
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        # Create PillarFeatureNetOld layers
        num_filters = [num_input_features] + list(num_filters)
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            if i < len(num_filters) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, use_norm, last_layer=last_layer))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + pc_range[0]
        self.y_offset = self.vy / 2 + pc_range[1]

    def forward(self, features, num_voxels, coors):
        device = features.device

        dtype = features.dtype
        # Find distance of x, y, and z from cluster center
        points_mean = features[:, :, :3].sum(
            dim=1, keepdim=True) / num_voxels.type_as(features).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean
        pp_min = features[:, :, 2:3].min(dim=1, keepdim=True)[0]
        pp_max = features[:, :, 2:3].max(dim=1, keepdim=True)[0]
        pp_height = pp_max - pp_min
        # Find distance of x, y, and z from pillar center
        f_height = torch.zeros_like(features[:, :, :1])
        f_height[:] = pp_height
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (
            coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (
            coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)
        features_radius = torch.norm(features[:, :, :2], p=2, dim=2, keepdim=True)
        features_radius = torch.cat([features_radius, features[:, :, 2:]], dim=2)
        # Combine together feature decorations
        features_ls = [features_radius, f_cluster, f_center, f_height]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


@register_middle
class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=64,
                 num_filters_down1=[64],
                 num_filters_down2=[64, 64],
                 name='SpMiddle2K'):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's    将学习到的特征从稠密张量转换为稀疏伪图像。这替换SECOND的

        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape           #[1,1,400,400,64]
        self.ny = output_shape[2]                  #400
        self.nx = output_shape[3]                  #400
        self.nchannels = num_input_features        #64

    # def forward(self, voxel_features,radar_voxel_features, coords,radar_coords, batch_size):  #batch_size=3
    def forward(self, voxel_features, coords, batch_size):  # batch_size=3
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample       #为这个sample创建画布
            canvas = torch.zeros(         #shape(64,160000)   400*400?
                self.nchannels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # radar_canvas = torch.zeros(  # shape(64,160000)   400*400?
            #     self.nchannels,
            #     self.nx * self.ny,
            #     dtype=radar_voxel_features.dtype,
            #     device=radar_voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt       #分离出batch_mask=0,1,2的coordinator
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]    #坐标(162*400+204)
            indices = indices.type(torch.long)      #柱子索引 5897
            voxels = voxel_features[batch_mask, :]     #5897    5897/38093= 0.15           21235
            voxels = voxels.t()        #转至  #[5897,64]

            # radar_batch_mask = radar_coords[:, 0] == batch_itt
            # radar_this_coords = radar_coords[radar_batch_mask, :]
            # radar_indices = radar_this_coords[:,2] * self.nx + radar_this_coords[:, 3]
            # radar_indices = radar_indices.type(torch.long)
            # radar_voxels = radar_voxel_features[radar_batch_mask, :]
            # radar_voxels = radar_voxels.t()

            # radar_canvas[:,radar_indices] = radar_voxels
            # radar_canvas = radar_canvas.view(1,self.nchannels, self.ny,
            #                              self.nx)
            # torch.set_printoptions(threshold=200000)
            # print(radar_canvas)

            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # radar_canvas=radar_canvas.cuda()
            # conv2d_1= nn.Conv2d(in_channels=64,out_channels=1,kernel_size=1,stride=1,bias=False)
            # conv2d_2 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1,bias=False)
            # conv2d_3 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5, stride=1, padding=2,bias=False)
            # conv2d_1=conv2d_1.cuda()
            # conv2d_2 = conv2d_2.cuda()
            # conv2d_3 = conv2d_3.cuda()
            # radar_canvas_1= conv2d_1(radar_canvas)
            # radar_canvas_2 = conv2d_2(radar_canvas)
            # radar_canvas_3 = conv2d_3(radar_canvas)
            # radar_attention_matrix = torch.add(radar_canvas_1,radar_canvas_2)
            # radar_attention_matrix = torch.add(radar_attention_matrix,radar_canvas_3)
            # # print(radar_attention_matrix)
            # radar_attention_matrix_1=np.squeeze(radar_attention_matrix)
            # one = torch.ones_like(radar_attention_matrix_1)
            # radar_attention_matrix_1 = torch.abs(radar_attention_matrix_1)
            # radar_attention_matrix = torch.add(radar_attention_matrix_1, one)

            # Now scatter the blob back to the canvas. 现在将斑点分散回画布
            canvas[:, indices] = voxels
            # canvas=canvas.view(self.nchannels, self.ny,
            # #                              self.nx)
            # print(canvas)


            # canvas = torch.mul(radar_attention_matrix,canvas)

            # Append to a list for later stacking.  追加到一个列表以供以后堆叠。
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols) 叠加到3-dim张量
        batch_canvas = torch.stack(batch_canvas, 0)        #[3,64,160000] 3是batch

        # Undo the column stacking to final 4-dim tensor 撤销列叠加到最终的4-dim张量
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny,
                                         self.nx)                    #[3,64,400,400]
        return batch_canvas
