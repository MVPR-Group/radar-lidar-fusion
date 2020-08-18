import copy
import json
import os

from pathlib import Path
import pickle
import shutil
import time
import re
import fire
import numpy as np
import torch
from google.protobuf import text_format
import sys
sys.path.append('/home/lichao/second-master_D8r')
import second.data.kitti_common as kitti
import torchplus
from second.builder import target_assigner_builder, voxel_builder
from second.core import box_np_ops
from second.data.preprocess import merge_second_batch, merge_second_batch_multigpu
from second.protos import pipeline_pb2
from second.pytorch.builder import (box_coder_builder, input_reader_builder,
                                    lr_scheduler_builder, optimizer_builder,
                                    second_builder)
from second.utils.log_tool import SimpleModelLog
from second.utils.progress_bar import ProgressBar
import psutil

def filter_param_dict(state_dict: dict, include: str=None, exclude: str=None):
    assert isinstance(state_dict, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    res_dict = {}
    for k, p in state_dict.items():
        if include_re is not None:
            if include_re.match(k) is None:
                continue
        if exclude_re is not None:
            if exclude_re.match(k) is not None:
                continue
        res_dict[k] = p
    return res_dict

input_cfg = config.train_input_reader
eval_input_cfg = config.eval_input_reader
model_cfg = config.model.second
train_cfg = config.train_config

net = build_network(model_cfg, measure_time).to(device)
    # if train_cfg.enable_mixed_precision:
model_dict = net.state_dict()
pretrained_path="/home/lichao/model_lowa_attention_D2_filter716/voxelnet-140650.tckpt"
pretrained_dict = torch.load(pretrained_path)
        # pretrained_dict = filter_param_dict(pretrained_dict, pretrained_include, pretrained_exclude)
new_pretrained_dict = {}
        # for k, v in pretrained_dict.items():
        #     if k in model_dict and v.shape == model_dict[k].shape:
        #         new_pretrained_dict[k] = v
print("Load pretrained parameters:")
for k, v in new_pretrained_dict.items(): #加载数据
        print(k, v.shape)