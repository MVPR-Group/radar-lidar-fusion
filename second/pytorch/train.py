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


def example_convert_to_torch(example, dtype=torch.float32,
                             device=None) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels","radar_voxels","anchors", "reg_targets", "reg_weights", "bev_map", "importance"
    ]
    for k, v in example.items():                         #数据类型转换
        if k in float_names:
            # slow when directly provide fp32 data with dtype=torch.half
            example_torch[k] = torch.tensor(   #将数据换成GPU类型
                v, dtype=torch.float32, device=device).to(dtype)
        elif k in ["coordinates","radar_coordinates", "labels", "num_points","radar_num_points"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.int32, device=device)
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.tensor(
                v, dtype=torch.uint8, device=device)
        elif k == "calib":
            calib = {}
            for k1, v1 in v.items():
                calib[k1] = torch.tensor(
                    v1, dtype=dtype, device=device).to(dtype)
            example_torch[k] = calib
        elif k == "num_voxels":
            example_torch[k] = torch.tensor(v)
        elif k == "num_radar_voxels":
            example_torch[k] = torch.tensor(v)
        else:
            example_torch[k] = v
    return example_torch


def build_network(model_cfg, measure_time=False):
    voxel_generator = voxel_builder.build(model_cfg.voxel_generator)
    bv_range = voxel_generator.point_cloud_range[[0, 1, 3, 4]]         #[-50,50,-50,50]
    box_coder = box_coder_builder.build(model_cfg.box_coder)
    target_assigner_cfg = model_cfg.target_assigner #对10个类的大小等config进行分配
    target_assigner = target_assigner_builder.build(target_assigner_cfg,
                                                    bv_range, box_coder)
    box_coder.custom_ndim = target_assigner._anchor_generators[0].custom_ndim
    net = second_builder.build(
        model_cfg, voxel_generator, target_assigner, measure_time=measure_time)
    return net
#net:
# VoxelNet(
#   (voxel_feature_extractor): PillarFeatureNet(
#     (pfn_layers): ModuleList(
#       (0): PFNLayer(
#         (linear): DefaultArgLayer(in_features=9, out_features=64, bias=False)
#         (norm): DefaultArgLayer(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#       )
#     )
#   )
#   (middle_feature_extractor): PointPillarsScatter()
#   (rpn): RPNV2(
#     (blocks): ModuleList(
#       (0): Sequential(
#         (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
#         (1): DefaultArgLayer(64, 64, kernel_size=(3, 3), stride=(2, 2), bias=False)
#         (2): DefaultArgLayer(64, eps=0·   .001, momentum=0.01, affine=True, track_running_stats=True)
#         (3): ReLU()
#         (4): DefaultArgLayer(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (5): DefaultArgLayer(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (6): ReLU()
#         (7): DefaultArgLayer(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (8): DefaultArgLayer(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (9): ReLU()
#         (10): DefaultArgLayer(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (11): DefaultArgLayer(64, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (12): ReLU()
#       )
#       (1): Sequential(
#         (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
#         (1): DefaultArgLayer(64, 128, kernel_size=(3, 3), stride=(2, 2), bias=False)
#         (2): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (3): ReLU()
#         (4): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (5): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (6): ReLU()
#         (7): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (8): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (9): ReLU()
#         (10): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (11): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (12): ReLU()
#         (13): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (14): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (15): ReLU()
#         (16): DefaultArgLayer(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (17): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (18): ReLU()
#       )
#       (2): Sequential(
#         (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
#         (1): DefaultArgLayer(128, 256, kernel_size=(3, 3), stride=(2, 2), bias=False)
#         (2): DefaultArgLayer(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (3): ReLU()
#         (4): DefaultArgLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (5): DefaultArgLayer(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (6): ReLU()
#         (7): DefaultArgLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (8): DefaultArgLayer(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (9): ReLU()
#         (10): DefaultArgLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (11): DefaultArgLayer(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (12): ReLU()
#         (13): DefaultArgLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (14): DefaultArgLayer(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (15): ReLU()
#         (16): DefaultArgLayer(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (17): DefaultArgLayer(256, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (18): ReLU()
#       )
#     )
#     (deblocks): ModuleList(
#       (0): Sequential(
#         (0): DefaultArgLayer(64, 128, kernel_size=(4, 4), stride=(4, 4), bias=False)
#         (1): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (2): ReLU()
#       )
#       (1): Sequential(
#         (0): DefaultArgLayer(128, 128, kernel_size=(2, 2), stride=(2, 2), bias=False)
#         (1): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (2): ReLU()
#       )
#       (2): Sequential(
#         (0): DefaultArgLayer(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (1): DefaultArgLayer(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#         (2): ReLU()
#       )
#     )
#     (conv_cls): Conv2d(384, 200, kernel_size=(1, 1), stride=(1, 1))
#     (conv_box): Conv2d(384, 140, kernel_size=(1, 1), stride=(1, 1))
#     (conv_dir_cls): Conv2d(384, 40, kernel_size=(1, 1), stride=(1, 1))
#   )
#   (rpn_acc): Accuracy()
#   (rpn_precision): Precision()
#   (rpn_recall): Recall()
#   (rpn_metrics): PrecisionRecall()
#   (rpn_cls_loss): Scalar()
#   (rpn_loc_loss): Scalar()
#   (rpn_total_loss): Scalar()
# )

def _worker_init_fn(worker_id):
    time_seed = np.array(time.time(), dtype=np.int32)
    np.random.seed(time_seed + worker_id)
    print(f"WORKER {worker_id} seed:", np.random.get_state()[1][0])

def freeze_params(params: dict, include: str=None, exclude: str=None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    remain_params = []
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                continue 
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                continue 
        remain_params.append(p)
    return remain_params

def freeze_params_v2(params: dict, include: str=None, exclude: str=None):
    assert isinstance(params, dict)
    include_re = None
    if include is not None:
        include_re = re.compile(include)
    exclude_re = None
    if exclude is not None:
        exclude_re = re.compile(exclude)
    for k, p in params.items():
        if include_re is not None:
            if include_re.match(k) is not None:
                p.requires_grad = False
        if exclude_re is not None:
            if exclude_re.match(k) is None:
                p.requires_grad = False

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


def train(config_path,
          model_dir,
          pretrained_path=None,
          result_path=None,
          create_folder=False,
          display_step=50,
          summary_step=5,

          pretrained_include=None,
          pretrained_exclude=None,
          freeze_include=None,
          freeze_exclude=None,
          multi_gpu=False,
          measure_time=False,
          resume=False):
    """train a VoxelNet model specified by a config file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_dir = str(Path(model_dir).resolve())
    if create_folder:
        if Path(model_dir).exists():
            model_dir = torchplus.train.create_folder(model_dir)
    model_dir = Path(model_dir)
    if not resume and model_dir.exists():
        path= os.path.join(str(Path(model_dir)))
        shutil.rmtree(path)
        # raise ValueError("model dir exists and you don't specify resume.")
    model_dir.mkdir(parents=True, exist_ok=True)
    if result_path is None:
        result_path = model_dir / 'results'
    config_file_bkp = "pipeline.config"
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)
    with (model_dir / config_file_bkp).open("w") as f:
        f.write(proto_str)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, measure_time).to(device)
    # if train_cfg.enable_mixed_precision:
    #     net.half()
    #     net.metrics_to_float()
    #     net.convert_norm_to_float(net)
    target_assigner = net.target_assigner   #########
    voxel_generator = net.voxel_generator   #########
    print("num parameters:", len(list(net.parameters())))  #66    72
    torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    if pretrained_path is not None:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = filter_param_dict(pretrained_dict, pretrained_include, pretrained_exclude)
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                new_pretrained_dict[k] = v        
        print("Load pretrained parameters:")
        for k, v in new_pretrained_dict.items(): #加载数据
            print(k, v.shape)
        model_dict.update(new_pretrained_dict) 
        net.load_state_dict(model_dict)
        freeze_params_v2(dict(net.named_parameters()), freeze_include, freeze_exclude)
        net.clear_global_step()
        net.clear_metrics()
    if multi_gpu:
        net_parallel = torch.nn.DataParallel(net)
    else:
        net_parallel = net
    optimizer_cfg = train_cfg.optimizer
    loss_scale = train_cfg.loss_scale_factor
    fastai_optimizer = optimizer_builder.build(
        optimizer_cfg,
        net,
        mixed=False,
        loss_scale=loss_scale)
    if loss_scale < 0:
        loss_scale = "dynamic"
    if train_cfg.enable_mixed_precision:
        max_num_voxels = input_cfg.preprocess.max_number_of_voxels * input_cfg.batch_size
        assert max_num_voxels < 65535, "spconv fp16 training only support this"   #多GPU训练
        from apex import amp
        net, amp_optimizer = amp.initialize(net, fastai_optimizer,
                                        opt_level="O2",
                                        keep_batchnorm_fp32=True,
                                        loss_scale=loss_scale
                                        )
        net.metrics_to_float()
    else:
        amp_optimizer = fastai_optimizer
    torchplus.train.try_restore_latest_checkpoints(model_dir,
                                                   [fastai_optimizer])
    lr_scheduler = lr_scheduler_builder.build(optimizer_cfg, amp_optimizer,
                                              train_cfg.steps)
    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    if multi_gpu:
        num_gpu = torch.cuda.device_count()
        print(f"MULTI-GPU: use {num_gpu} gpu")
        collate_fn = merge_second_batch_multigpu
    else:
        collate_fn = merge_second_batch
        num_gpu = 1

    ######################
    # PREPARE INPUT
    ######################
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=multi_gpu)                         #加载完训练数据
    eval_dataset = input_reader_builder.build(
        eval_input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)

    dataloader = torch.utils.data.DataLoader(
        dataset,                                      #dataset : 即上面自定义的dataset.
        batch_size=input_cfg.batch_size * num_gpu,    #3
        shuffle=True,                                 #打乱数据
        num_workers=input_cfg.preprocess.num_workers * num_gpu,   #使用多进程加载的进程数，0代表不使用多进程
        pin_memory=False,                             #是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
        collate_fn=collate_fn,                        # 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
        worker_init_fn=_worker_init_fn,               #
        drop_last=not multi_gpu)                      #dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_input_cfg.batch_size, # only support multi-gpu train
        shuffle=False,
        num_workers=eval_input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    ######################
    # TRAINING
    ######################
    model_logging = SimpleModelLog(model_dir)
    model_logging.open()
    model_logging.log_text(proto_str + "\n", 0, tag="config")
    start_step = net.get_global_step()
    total_step = train_cfg.steps
    t = time.time()
    steps_per_eval = train_cfg.steps_per_eval
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    amp_optimizer.zero_grad()
    step_times = []
    step = start_step
    try:
        while True:
            if clear_metrics_every_epoch:
                net.clear_metrics()
            for example in dataloader:    #example dict
                lr_scheduler.step(net.get_global_step())
                time_metrics = example["metrics"]
                example.pop("metrics")    #移除metrics
                example_torch = example_convert_to_torch(example, float_dtype) #数据类型转换voxel,num_point,coordinates,num_voxel,anchors,gt_name,labels,reg_targets,importance,meatadata

                batch_size = example["anchors"].shape[0]        #  3          #为什么batch_size要看anchor的shape [3,50000,7]

                ret_dict = net_parallel(example_torch)    #voxelnet.py
                cls_preds = ret_dict["cls_preds"]
                loss = ret_dict["loss"].mean()
                cls_loss_reduced = ret_dict["cls_loss_reduced"].mean()
                loc_loss_reduced = ret_dict["loc_loss_reduced"].mean()
                cls_pos_loss = ret_dict["cls_pos_loss"].mean()
                cls_neg_loss = ret_dict["cls_neg_loss"].mean()
                loc_loss = ret_dict["loc_loss"]
                cls_loss = ret_dict["cls_loss"]
                
                cared = ret_dict["cared"]
                labels = example_torch["labels"]
                if train_cfg.enable_mixed_precision:
                    with amp.scale_loss(loss, amp_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()   #标量
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)  #梯度裁剪，梯度最大范数10
                amp_optimizer.step()
                amp_optimizer.zero_grad()
                net.update_global_step()
                net_metrics = net.update_metrics(cls_loss_reduced,
                                                 loc_loss_reduced, cls_preds,
                                                 labels, cared)

                step_time = (time.time() - t)
                step_times.append(step_time)
                t = time.time()
                metrics = {}
                num_pos = int((labels > 0)[0].float().sum().cpu().numpy())
                num_neg = int((labels == 0)[0].float().sum().cpu().numpy())
                if 'anchors_mask' not in example_torch:
                    num_anchors = example_torch['anchors'].shape[1]      #50000
                else:
                    num_anchors = int(example_torch['anchors_mask'][0].sum())
                global_step = net.get_global_step()

                if global_step % display_step == 0:
                    if measure_time:
                        for name, val in net.get_avg_time_dict().items():
                            print(f"avg {name} time = {val * 1000:.3f} ms")

                    loc_loss_elem = [
                        float(loc_loss[:, :, i].sum().detach().cpu().numpy() /
                              batch_size) for i in range(loc_loss.shape[-1])
                    ]
                    metrics["runtime"] = {
                        "step": global_step,
                        "steptime": np.mean(step_times),
                    }
                    metrics["runtime"].update(time_metrics[0])
                    step_times = []
                    metrics.update(net_metrics)
                    metrics["loss"]["loc_elem"] = loc_loss_elem
                    metrics["loss"]["cls_pos_rt"] = float(
                        cls_pos_loss.detach().cpu().numpy())
                    metrics["loss"]["cls_neg_rt"] = float(
                        cls_neg_loss.detach().cpu().numpy())
                    if model_cfg.use_direction_classifier:
                        dir_loss_reduced = ret_dict["dir_loss_reduced"].mean()
                        metrics["loss"]["dir_rt"] = float(
                            dir_loss_reduced.detach().cpu().numpy())

                    metrics["misc"] = {
                        "num_vox": int(example_torch["voxels"].shape[0]),
                        "num_pos": int(num_pos),
                        "num_neg": int(num_neg),
                        "num_anchors": int(num_anchors),
                        "lr": float(amp_optimizer.lr),
                        "mem_usage": psutil.virtual_memory().percent,
                    }
                    model_logging.log_metrics(metrics, global_step)

                if global_step % steps_per_eval == 0:
                    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                                net.get_global_step())
                    net.eval()                                                             #训练完train_datasets之后，model要来测试样本了。在model(test_datasets)之前，需要加上model.eval().
                                                                                          # 否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有batch normalization层所带来的的性质。

                    result_path_step = result_path / f"step_{net.get_global_step()}"
                    result_path_step.mkdir(parents=True, exist_ok=True)
                    model_logging.log_text("#################################",
                                        global_step)
                    model_logging.log_text("# EVAL", global_step)
                    model_logging.log_text("#################################",
                                        global_step)
                    model_logging.log_text("Generate output labels...", global_step)
                    t = time.time()
                    detections = []
                    prog_bar = ProgressBar()
                    net.clear_timer()
                    prog_bar.start((len(eval_dataset) + eval_input_cfg.batch_size - 1)
                                // eval_input_cfg.batch_size)
                    for example in iter(eval_dataloader):
                        example = example_convert_to_torch(example, float_dtype)
                        detections += net(example)
                        prog_bar.print_bar()

                    sec_per_ex = len(eval_dataset) / (time.time() - t)
                    model_logging.log_text(
                        f'generate label finished({sec_per_ex:.2f}/s). start eval:',
                        global_step)
                    result_dict = eval_dataset.dataset.evaluation(
                        detections, str(result_path_step))
                    for k, v in result_dict["results"].items():
                        model_logging.log_text("Evaluation {}".format(k), global_step)
                        model_logging.log_text(v, global_step)
                    model_logging.log_metrics(result_dict["detail"], global_step)
                    with open(result_path_step / "result.pkl", 'wb') as f:
                        pickle.dump(detections, f)
                    net.train()
                step += 1
                if step >= total_step:
                    break
            if step >= total_step:
                break
    except Exception as e:
        print(json.dumps(example["metadata"], indent=2))
        model_logging.log_text(str(e), step)
        model_logging.log_text(json.dumps(example["metadata"], indent=2), step)
        torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                    step)
        raise e
    finally:
        model_logging.close()
    torchplus.train.save_models(model_dir, [net, amp_optimizer],
                                net.get_global_step())


def evaluate(config_path,
             ckpt_path,
             model_dir=None,
             result_path=None,
             # pretrained_path=None,
             # pretrained_include=None,
             # pretrained_exclude=None,
             measure_time=False,
             batch_size=None,
             **kwargs):
    """Don't support pickle_result anymore. if you want to generate kitti label file,
    please use kitti_anno_to_label_file and convert_detection_to_kitti_annos
    in second.data.kitti_dataset.
    """
    assert len(kwargs) == 0
    model_dir = str(Path(model_dir).resolve())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_name = 'eval_results'
    if result_path is None:
        model_dir = Path(model_dir)
        result_path = model_dir / result_name
    else:
        result_path = Path(result_path)
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to eval with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path

    input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, measure_time=measure_time).to(device)
    if train_cfg.enable_mixed_precision:
        net.half()
        print("half inference!")
        net.metrics_to_float()
        net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator

    if ckpt_path is None:
        assert model_dir is not None
        torchplus.train.try_restore_latest_checkpoints(model_dir, [net])
    else:
        torchplus.train.restore(ckpt_path, net)

    # #########################################################################################
    # model_dict = net.state_dict()
    # pretrained_dict = torch.load(ckpt_path)
    # pretrained_dict = filter_param_dict(pretrained_dict, pretrained_include, pretrained_exclude)
    # new_pretrained_dict = {}
    # for k, v in pretrained_dict.items():
    #     if k in model_dict and v.shape == model_dict[k].shape:
    #         new_pretrained_dict[k] = v
    # print("Load pretrained parameters:")
    # for k, v in new_pretrained_dict.items():  # 加载数据
    #     print(k, v.shape)
    # model_dict.update(new_pretrained_dict)
    # net.load_state_dict(model_dict)
    # freeze_params_v2(dict(net.named_parameters()), freeze_include, freeze_exclude)
    # net.clear_global_step()
    # net.clear_metrics()


    # if multi_gpu:
    #     net_parallel = torch.nn.DataParallel(net)
    # else:
    # net = net




    batch_size = batch_size or input_cfg.batch_size
    eval_dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=False,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=input_cfg.preprocess.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch)

    if train_cfg.enable_mixed_precision:
        float_dtype = torch.float16
    else:
        float_dtype = torch.float32

    net.eval()
    result_path_step = result_path / f"step_{net.get_global_step()}"
    result_path_step.mkdir(parents=True, exist_ok=True)
    t = time.time()
    detections = []
    print("Generate output labels...")
    bar = ProgressBar()
    bar.start((len(eval_dataset) + batch_size - 1) // batch_size)
    prep_example_times = []
    prep_times = []
    t2 = time.time()

    for example in iter(eval_dataloader):
        if measure_time:
            prep_times.append(time.time() - t2)
            torch.cuda.synchronize()
            t1 = time.time()
        example = example_convert_to_torch(example, float_dtype)
        if measure_time:
            torch.cuda.synchronize()
            prep_example_times.append(time.time() - t1)
        with torch.no_grad():                  #不计算倒数，反向传播更新
            detections += net(example)
        bar.print_bar()
        if measure_time:
            t2 = time.time()

    sec_per_example = len(eval_dataset) / (time.time() - t)
    print(f'generate label finished({sec_per_example:.2f}/s). start eval:')
    if measure_time:
        print(
            f"avg example to torch time: {np.mean(prep_example_times) * 1000:.3f} ms"
        )
        print(f"avg prep time: {np.mean(prep_times) * 1000:.3f} ms")
    for name, val in net.get_avg_time_dict().items():
        print(f"avg {name} time = {val * 1000:.3f} ms")
    with open(result_path_step / "result.pkl", 'wb') as f:
        pickle.dump(detections, f)
    result_dict = eval_dataset.dataset.evaluation(detections,
                                                  str(result_path_step))
    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print("Evaluation {}".format(k))
            print(v)

def helper_tune_target_assigner(config_path, target_rate=None, update_freq=200, update_delta=0.01, num_tune_epoch=5):
    """get information of target assign to tune thresholds in anchor generator.
    """    
    if isinstance(config_path, str):
        # directly provide a config object. this usually used
        # when you want to train with several different parameters in
        # one script.
        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
    else:
        config = config_path
        proto_str = text_format.MessageToString(config, indent=2)

    input_cfg = config.train_input_reader
    eval_input_cfg = config.eval_input_reader
    model_cfg = config.model.second
    train_cfg = config.train_config

    net = build_network(model_cfg, False)
    # if train_cfg.enable_mixed_precision:
    #     net.half()
    #     net.metrics_to_float()
    #     net.convert_norm_to_float(net)
    target_assigner = net.target_assigner
    voxel_generator = net.voxel_generator
    dataset = input_reader_builder.build(
        input_cfg,
        model_cfg,
        training=True,
        voxel_generator=voxel_generator,
        target_assigner=target_assigner,
        multi_gpu=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn,
        drop_last=False)
    
    class_count = {}
    anchor_count = {}
    class_count_tune = {}
    anchor_count_tune = {}
    for c in target_assigner.classes:
        class_count[c] = 0
        anchor_count[c] = 0
        class_count_tune[c] = 0
        anchor_count_tune[c] = 0


    step = 0
    classes = target_assigner.classes
    if target_rate is None:
        num_tune_epoch = 0
    for epoch in range(num_tune_epoch):
        for example in dataloader:
            gt_names = example["gt_names"]
            for name in gt_names:
                class_count_tune[name] += 1
            
            labels = example['labels']
            for i in range(1, len(classes) + 1):
                anchor_count_tune[classes[i - 1]] += int(np.sum(labels == i))
            if target_rate is not None:
                for name, rate in target_rate.items():
                    if class_count_tune[name] > update_freq:
                        # calc rate
                        current_rate = anchor_count_tune[name] / class_count_tune[name]
                        if current_rate > rate:
                            target_assigner._anchor_generators[classes.index(name)].match_threshold += update_delta
                            target_assigner._anchor_generators[classes.index(name)].unmatch_threshold += update_delta
                        else:
                            target_assigner._anchor_generators[classes.index(name)].match_threshold -= update_delta
                            target_assigner._anchor_generators[classes.index(name)].unmatch_threshold -= update_delta
                        anchor_count_tune[name] = 0
                        class_count_tune[name] = 0
            step += 1
    for c in target_assigner.classes:
        class_count[c] = 0
        anchor_count[c] = 0
    total_voxel_gene_time = 0
    count = 0

    for example in dataloader:
        gt_names = example["gt_names"]
        total_voxel_gene_time += example["metrics"][0]["voxel_gene_time"]
        count += 1

        for name in gt_names:
            class_count[name] += 1
        
        labels = example['labels']
        for i in range(1, len(classes) + 1):
            anchor_count[classes[i - 1]] += int(np.sum(labels == i))
    print("avg voxel gene time", total_voxel_gene_time / count)

    print(json.dumps(class_count, indent=2))
    print(json.dumps(anchor_count, indent=2))
    if target_rate is not None:
        for ag in target_assigner._anchor_generators:
            if ag.class_name in target_rate:
                print(ag.class_name, ag.match_threshold, ag.unmatch_threshold)

def mcnms_parameters_search(config_path,
          model_dir,
          preds_path):
    pass

def delete_dir(root):
    dirlist = os.listdir(root)   # 返回dir文件夹里的所有文件列表
    for f in dirlist:
        filepath = root + '\\' + f    # 路径与文件名拼接成完整的路径
        print(filepath)
        if os.path.isdir(filepath):      # 如果该文件是个文件夹
            delete_dir(filepath)        # 递归调用函数，将该文件夹内的文件删掉
            os.rmdir(filepath)         # 把文件夹删掉
        else:
            os.remove(filepath)      # 如果该文件不是文件夹，直接删除
    os.rmdir(root)   # 最后还需要把root删掉


if __name__ == '__main__':
    fire.Fire()
