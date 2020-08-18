import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet

from torchplus.nn import Empty, GroupNorm, Sequential
from torchplus.tools import change_default_args

REGISTERED_RPN_CLASSES = {}

def register_rpn(cls, name=None):
    global REGISTERED_RPN_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_RPN_CLASSES, f"exist class: {REGISTERED_RPN_CLASSES}"
    REGISTERED_RPN_CLASSES[name] = cls
    return cls

def get_rpn_class(name):
    global REGISTERED_RPN_CLASSES
    assert name in REGISTERED_RPN_CLASSES, f"available class: {REGISTERED_RPN_CLASSES}"
    return REGISTERED_RPN_CLASSES[name]

@register_rpn
class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """deprecated. exists for checkpoint backward compilability (SECOND v1.0)
        """
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        upsample_strides = [
            np.round(u).astype(np.int64) for u in upsample_strides
        ]
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(
                layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(
                np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_features, num_filters[0], 3,
                stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters),
                num_anchor_per_loc * num_direction_bins, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
                1)

    def forward(self, x):
        # t = time.time()
        # torch.cuda.synchronize()

        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        if self._use_rc_net:
            rc_preds = self.conv_rc(x)
            rc_preds = rc_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["rc_preds"] = rc_preds
        # torch.cuda.synchronize()
        # print("rpn forward time", time.time() - t)

        return ret_dict

class RPNNoHeadBase(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2, #10
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNNoHeadBase, self).__init__()
        self._layer_strides = layer_strides
        self._num_filters = num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = upsample_strides
        self._num_upsample_filters = num_upsample_filters
        self._num_input_features = num_input_features
        self._use_norm = use_norm
        self._use_groupnorm = use_groupnorm
        self._num_groups = num_groups
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(num_upsample_filters) == len(upsample_strides)
        self._upsample_start_idx = len(layer_nums) - len(upsample_strides)
        # self.attenconv1 = nn.Conv2d(64, 64, 1, 1)
        # self.attenconv2 = nn.Conv2d(64, 64, 3, 1,padding=1)
        # self.attenconv3 = nn.Conv2d(64, 64, 5, 1,padding=2)

        must_equal_list = []
        for i in range(len(upsample_strides)):
            must_equal_list.append(upsample_strides[i] / np.prod(
                layer_strides[:i + self._upsample_start_idx + 1]))
        for val in must_equal_list:
            assert val == must_equal_list[0]

        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)   #num_groups=32
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)# change_default_args函数意思  nn.Conv2d的参数 channels,output,卷积核（height_2,width_2）,stride默认为1，groups卷积核个数
            ConvTranspose2d = change_default_args(bias=False)(    #反卷积 #in_channels, out_channels, kernel_size, stride=1, padding=0,  output_padding=0, groups=1, bias=True, dilation=1)
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)
  ###################################################
        if use_norm:
            if use_groupnorm:
                BatchNorm2d_radar = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)   #num_groups=32
            else:
                BatchNorm2d_radar = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d_radar = change_default_args(bias=False)(nn.Conv2d)# change_default_args函数意思  nn.Conv2d的参数 channels,output,卷积核（height_2,width_2）,stride默认为1，groups卷积核个数
            ConvTranspose2d_radar = change_default_args(bias=False)(    #反卷积 #in_channels, out_channels, kernel_size, stride=1, padding=0,  output_padding=0, groups=1, bias=True, dilation=1)
                nn.ConvTranspose2d)
        else:
            BatchNorm2d_radar = Empty
            Conv2d_radar = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d_radar = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        in_filters = [num_input_features, *num_filters[:-1]]
        blocks = []
        radar_blocks=[]
        deblocks = []
        radar_deblocks=[]
        # radar_block=[]

        # attenblock1s=[]
        # attenblock2s=[]
        # attenblock3s=[]

        for i, layer_num in enumerate(layer_nums):#layer_nums [3,5,5]
            block, num_out_filters = self._make_layer(         #num_out_filter=128        #到632行 make_layer函数
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i])
            blocks.append(block)
            # if layer_num==3:
            #     block,num_out_filters = self._make_layer(in_filters[i],
            #     num_filters[i],
            #     layer_num,
            #     stride=layer_strides[i])
            #     blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = upsample_strides[i - self._upsample_start_idx]  #<class 'list'>: [0.25, 0.5, 1.0]
                if stride >= 1:
                    stride = np.round(stride).astype(np.int64)
                    deblock = nn.Sequential(
                        ConvTranspose2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    deblock = nn.Sequential(
                        Conv2d(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                deblocks.append(deblock)
#########################################################################radar
        for i, layer_num in enumerate(layer_nums):#layer_nums [3,5,5]
            radar_block, num_out_filters = self._make_layer(         #num_out_filter=128
                in_filters[i],
                num_filters[i],
                layer_num,
                stride=layer_strides[i])
            radar_blocks.append(radar_block)
            # if layer_num==3:
            #     block,num_out_filters = self._make_layer(in_filters[i],
            #     num_filters[i],
            #     layer_num,
            #     stride=layer_strides[i])
            #     blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = upsample_strides[i - self._upsample_start_idx]  #<class 'list'>: [0.25, 0.5, 1.0]
                if stride >= 1:
                    stride = np.round(stride).astype(np.int64)
                    radar_deblock = nn.Sequential(
                        ConvTranspose2d_radar(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d_radar(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    radar_deblock = nn.Sequential(
                        Conv2d_radar(
                            num_out_filters,
                            num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride),
                        BatchNorm2d_radar(
                            num_upsample_filters[i -
                                                 self._upsample_start_idx]),
                        nn.ReLU(),
                    )
                radar_deblocks.append(radar_deblock)
        # attenblock1 = nn.Sequential(Conv2d(64,64,1,1))
        # attenblock2 = nn.Sequential(Conv2d(64, 64, 3, 1,padding=1))
        # attenblock3 = nn.Sequential(Conv2d(64, 64, 5, 1,padding=2))
        # attenblock1s.append(attenblock1)
        # attenblock2s.append(attenblock2)
        # attenblock3s.append(attenblock3)
        # radar_blocks=blocks
        # radar_deblocks=deblocks
        self._num_out_filters = num_out_filters    #256
        self.blocks = nn.ModuleList(blocks)
        self.radar_blocks=nn.ModuleList(radar_blocks)

        self.deblocks = nn.ModuleList(deblocks) #如果在构造函数__init__中用到list、tuple、dict等对象时，一定要思考是否应该用ModuleList或ParameterList代替。
        self.radar_deblocks = nn.ModuleList(radar_deblocks)   # 如果你想设计一个神经网络的层数作为输入传递。

        # self.attenconv1 = nn.Conv2d(384, 384, 1, 1)
        # self.BatchNorm2d1= nn.BatchNorm2d(384,eps=1e-3, momentum=0.01)
        # self.rule1 = nn.ReLU()
        # self.attenconv2 = nn.Conv2d(384, 384, 1, 1)
        # self.BatchNorm2d2=nn.BatchNorm2d(384,eps=1e-3, momentum=0.01)
        # self.rule2 = nn.ReLU()
        # self.attenconv3 = nn.Conv2d(384, 384, 1, 1)
        # self.BatchNorm2d3 = nn.BatchNorm2d(384,eps=1e-3, momentum=0.01)
        # self.rule3 = nn.ReLU()
        # self.attenconv4 = nn.Conv2d(384, 384, 1, 1)
        # self.BatchNorm2d4 = nn.BatchNorm2d(384,eps=1e-3, momentum=0.01)
        # self.rule4 = nn.ReLU()
        # zero = torch.Tensor(np.zeros([384,50,50]))
        # self.w = nn.Parameter(zero)




    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        raise NotImplementedError

    def forward(self,x):
        if x.shape[0]==6:
            x, x_radar = x.split(3, 0)        #train batch_size=3
        if x.shape[0]==2:
            x, x_radar = x.split(1, 0)        #eval batch_size=1
        # print(x.shape[0])
        ups = []
        radar_ups=[]
        stage_outputs = []
        # n=2500
        # channel=384
        # print("radar1", np.sum(x_radar.cpu().detach().numpy() != 0.0))
        # print("lidar1", np.sum(x.cpu().detach().numpy() != 0.0))
        # for i in range(len(self.blocks)):
        #     if i==0:
        #
        #         x_radar = self.blocks[0](x_radar)
        #         # print("radar2", np.sum(x_radar.cpu().detach().numpy() != 0.0))
        #         q = self.attenconv1(x_radar)
        #         q=torch.reshape(q,[-1,n,channel])
        #         # x_radar_2 = self.attenconv2(x_radar)
        #         # x_radar_3 = self.attenconv3(x_radar)
        #         # radar_attention_matrix = torch.add(x_radar_1, x_radar_2)
        #         # radar_attention_matrix = torch.add(radar_attention_matrix, x_radar_3)
        #         # print("radar_attention_matrix",np.sum(radar_attention_matrix.cpu().detach().numpy()>0.0))
        #         # one = torch.ones_like(radar_attention_matrix)
        #         # radar_attention_matrix = torch.add(radar_attention_matrix,one)
        #         continue
        #     if i==1:
        #
        #         x = self.blocks[i](x)   #[64,200,200][128,100,100] [256,50,50]      #跳到common.py 85行 forward
        #         # print("lidar2", np.sum(x.cpu().detach().numpy() != 0.0))
        #         k=self.attenconv2(x)
        #         v=self.attenconv3(x)
        #         k=torch.reshape(k,[-1,n,channel]).permute(0,2,1)
        #         v=torch.reshape(v,[-1,n,channel])
        #         for j in range(len(q)):
        #             radar_attention_matrix = torch.mm(q[j], k[j])
        #         # radar_attention_matrix=torch.nn.Softmax(radar_attention_matrix,dim=1)
        #         # x = torch.add(radar_attention_matrix, x)
        #         radar_attention_matrix.permute(0,2,1)
        #         x= torch.mul(radar_attention_matrix,v)
        #         v= torch.reshape(x,[-1,200,200,channel])
        #         x= self.attenconv4(x)
        #
        #
        #     elif i>1:
        #         x=self.blocks[i](x)
        #     stage_outputs.append(x)
        #     i=i-1
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stage_outputs.append(x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))     #[128,50,50],[128,50,50],[128,50,50]
        for i in range(len(self.radar_blocks)):
            x_radar = self.radar_blocks[i](x_radar)
            stage_outputs.append(x_radar)
            if i - self._upsample_start_idx >= 0:
                radar_ups.append(self.radar_deblocks[i - self._upsample_start_idx](x_radar))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1)   #[384,50,50]
        if len(radar_ups)>0:
            x_radar = torch.cat(radar_ups,dim=1)

        # q = self.attenconv1(x_radar)
        # q = self.BatchNorm2d1(q)
        # q = self.rule1(q)
        # # print("q", np.sum(q.cpu().detach().numpy() < 0.0))
        # k = self.attenconv2(x)
        # k = self.BatchNorm2d2(k)
        # k = self.rule2(k)
        # # print("k", np.sum(k.cpu().detach().numpy() < 0.0))
        # v = self.attenconv3(x)
        # v = self.BatchNorm2d3(v)
        # v = self.rule3(v)
        # # print("v", np.sum(v.cpu().detach().numpy() < 0.0))
        # q = torch.reshape(q, [-1, channel, n])
        # k = torch.reshape(k, [-1, channel, n]).permute(0, 2, 1)
        # v=torch.reshape(v,[-1,channel, n])
        # radar_attention_matrix = torch.bmm(k, q)
        # # radar_attention_matrix = torch.mul(k, q)
        # m=nn.Softmax(dim=1)
        # radar_attention_matrix = m(radar_attention_matrix)
        # #radar_attention_matrix=radar_attention_matrix.permute(0, 2, 1)
        # o= torch.bmm(v,radar_attention_matrix)
        # # o = torch.mul(v, radar_attention_matrix)
        # o = torch.reshape(o, [-1, channel, 50, 50])#########################
        # o = self.attenconv4(o)
        # o = self.BatchNorm2d4(o)
        # o = self.rule4(o)
        # print(next(o.parameters()).is_cuda)
        # print(o.device)
        # print(self.w.device)
        # print("o", np.sum(o.cpu().detach().numpy() < 0.0))
        x=torch.add(x_radar,x)


        res = {}
        for i, up in enumerate(ups):
            res[f"up{i}"] = up
        for i, out in enumerate(stage_outputs):
            res[f"stage{i}"] = out
        res["out"] = x
        return res


class RPNBase(RPNNoHeadBase):
    def __init__(self,
                 use_norm=True,
                 num_class=2,      #num_class=10
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),           #64,128,256
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,         #64
                 num_anchor_per_loc=2,          #20
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """upsample_strides support float: [0.25, 0.5, 1]
        if upsample_strides < 1, conv2d will be used instead of convtranspose2d.
        """
        super(RPNBase, self).__init__(
            use_norm=use_norm,
            num_class=num_class,
            layer_nums=layer_nums,
            layer_strides=layer_strides,
            num_filters=num_filters,
            upsample_strides=upsample_strides,
            num_upsample_filters=num_upsample_filters,
            num_input_features=num_input_features,
            num_anchor_per_loc=num_anchor_per_loc,
            encode_background_as_zeros=encode_background_as_zeros,
            use_direction_classifier=use_direction_classifier,
            use_groupnorm=use_groupnorm,
            num_groups=num_groups,
            box_code_size=box_code_size,
            num_direction_bins=num_direction_bins,
            name=name)
        self._num_anchor_per_loc = num_anchor_per_loc
        self._num_direction_bins = num_direction_bins
        self._num_class = num_class
        self._use_direction_classifier = use_direction_classifier
        self._box_code_size = box_code_size

        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        if len(num_upsample_filters) == 0:
            final_num_filters = self._num_out_filters
        else:
            final_num_filters = sum(num_upsample_filters)
        self.conv_cls = nn.Conv2d(final_num_filters, num_cls, 1)  #final_num_filters=384,num_cls=200
        self.conv_box = nn.Conv2d(final_num_filters,                  #final_num_filter=384
                                  num_anchor_per_loc * box_code_size, 1)  # num_anchor_per_loc=20 ， box_code_size=7
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                final_num_filters, num_anchor_per_loc * num_direction_bins, 1)

    def forward(self, x):
        res = super().forward(x)     #[64,400,400]     ---->上采样后变成3个[128 50 50]??????
        x = res["out"]                #[384,50,50]
        box_preds = self.conv_box(x)               #140 50 50   Conv2d(384, 140, kernel_size=(1, 1), stride=(1, 1))
        cls_preds = self.conv_cls(x)               #（200,50,50）Conv2d(384, 200, kernel_size=(1, 1), stride=(1, 1))
        # [N, C, y(H), x(W)]
        C, H, W = box_preds.shape[1:]             #C=140,H=50,W=50
        box_preds = box_preds.view(-1, self._num_anchor_per_loc,              #[3,20,50,50,7]  _num_anchor_per_loc=20  _box_code_size=7
                                   self._box_code_size, H, W).permute(
                                       0, 1, 3, 4, 2).contiguous()
        cls_preds = cls_preds.view(-1, self._num_anchor_per_loc,              #[3,20,50,50,10]
                                   self._num_class, H, W).permute(              #_num_class=10
                                       0, 1, 3, 4, 2).contiguous()
        # box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        # cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()

        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)         #dir_cls_preds=torch.Size([3, 40, 50, 50]),x=  torch.Size([3, 384, 50, 50])
            dir_cls_preds = dir_cls_preds.view(             # dir_cls_preds=torch.Size([3, 20, 50, 50, 2])
                -1, self._num_anchor_per_loc, self._num_direction_bins, H,     #num_derection_bins
                W).permute(0, 1, 3, 4, 2).contiguous()
            # dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds  #torch.Size([3, 20, 50, 50, 2])
        return ret_dict


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

@register_rpn
class ResNetRPN(RPNBase):
    def __init__(self, *args, **kw):
        self.inplanes = -1
        super(ResNetRPN, self).__init__(*args, **kw)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # if zero_init_residual:
        for m in self.modules():
            if isinstance(m, resnet.Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, resnet.BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self.inplanes == -1:
            self.inplanes = self._num_input_features
        block = resnet.BasicBlock
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers), self.inplanes

@register_rpn
class RPNV2(RPNBase):
    def _make_layer(self, inplanes, planes, num_blocks, stride=1):     #inplanes=64,num_blocks=3,plane=64,stride=2
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),   #64 ，64 ,3,2
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):        #3个blocks
            block.add(Conv2d(planes, planes, 3, padding=1))   #
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes   #block Sequential(
#   (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
#   (1): DefaultArgLayer(64, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)
#   (2): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#   (3): ReLU()
#   (4): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (5): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#   (6): ReLU()
#   (7): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (8): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#   (9): ReLU()
#   (10): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (11): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#   (12): ReLU()
#   (13): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (14): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#   (15): ReLU()
#   (16): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#   (17): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#   (18): ReLU()
# )
#plane =128

@register_rpn
class RPNNoHead(RPNNoHeadBase):
    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        if self._use_norm:
            if self._use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=self._num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        block = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(inplanes, planes, 3, stride=stride),
            BatchNorm2d(planes),
            nn.ReLU(),
        )
        for j in range(num_blocks):
            block.add(Conv2d(planes, planes, 3, padding=1))
            block.add(BatchNorm2d(planes))
            block.add(nn.ReLU())

        return block, planes
