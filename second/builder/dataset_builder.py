# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Input reader builder.

Creates data sources for DetectionModels from an InputReader config. See
input_reader.proto for options.

Note: If users wishes to also use their own InputReaders with the Object
Detection configuration framework, they should define their own builder function
that wraps the build function.
"""

from second.protos import input_reader_pb2
from second.data.dataset import get_dataset_class
from second.data.preprocess import prep_pointcloud
from second.core import box_np_ops
import numpy as np
from second.builder import dbsampler_builder
from functools import partial
from second.utils.config_tool import get_downsample_factor

def build(input_reader_config,
          model_config,
          training,
          voxel_generator,
          target_assigner,
          multi_gpu=False):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(input_reader_config, input_reader_pb2.InputReader):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    prep_cfg = input_reader_config.preprocess
    dataset_cfg = input_reader_config.dataset  #kitti_info_path: "/home/lichao/v1.0-mini/infos_train.pkl"
                                                  # kitti_root_path: "/home/lichao/v1.0-mini"
                                                  # dataset_class_name: "NuScenesDataset"
    num_point_features = model_config.num_point_features  #4
    out_size_factor = get_downsample_factor(model_config)    #8
    assert out_size_factor > 0
    cfg = input_reader_config
    db_sampler_cfg = prep_cfg.database_sampler  #database_info_path: "/home/lichao/v1.0-mini/kitti_dbinfos_train.pkl"
                                            # sample_groups {   name_to_max_num     key: "car"     value: 30   } } global_random_rotation_range_per_object: 0.0# global_random_rotation_range_per_object: 0.0# rate: 1.0
    db_sampler = None
    if len(db_sampler_cfg.sample_groups) > 0 or db_sampler_cfg.database_info_path != "":  # enable sample
        db_sampler = dbsampler_builder.build(db_sampler_cfg)   #加载了gt_base的一些东西
    grid_size = voxel_generator.grid_size #[400,400]
    feature_map_size = grid_size[:2] // out_size_factor #[50,50]
    feature_map_size = [*feature_map_size, 1][::-1]   #[50,50]
    print("feature_map_size", feature_map_size)
    assert all([n != '' for n in target_assigner.classes]), "you must specify class_name in anchor_generators."
    dataset_cls = get_dataset_class(dataset_cfg.dataset_class_name) # NuScenesDataset
    assert dataset_cls.NumPointFeatures >= 3, "you must set this to correct value"
    assert dataset_cls.NumPointFeatures == num_point_features, "currently you need keep them same"
    prep_func = partial(        #pre_func partial 的功能：固定函数参数，返回一个新的函数。
        prep_pointcloud,        #data\preprocess.py
        root_path=dataset_cfg.kitti_root_path,
        voxel_generator=voxel_generator,   #VoxelGeneratorV2
        target_assigner=target_assigner,
        training=training,
        max_voxels=prep_cfg.max_number_of_voxels, #25000    eval 30000
        remove_outside_points=False,
        remove_unknown=prep_cfg.remove_unknown_examples,
        create_targets=training,
        shuffle_points=prep_cfg.shuffle_points,
        gt_rotation_noise=list(prep_cfg.groundtruth_rotation_uniform_noise),
        gt_loc_noise_std=list(prep_cfg.groundtruth_localization_noise_std),
        global_rotation_noise=list(prep_cfg.global_rotation_uniform_noise),
        global_scaling_noise=list(prep_cfg.global_scaling_uniform_noise),
        global_random_rot_range=list(
            prep_cfg.global_random_rotation_range_per_object),
        global_translate_noise_std=list(prep_cfg.global_translate_noise_std),
        db_sampler=db_sampler,
        num_point_features=dataset_cls.NumPointFeatures,
        anchor_area_threshold=prep_cfg.anchor_area_threshold,
        gt_points_drop=prep_cfg.groundtruth_points_drop_percentage,
        gt_drop_max_keep=prep_cfg.groundtruth_drop_max_keep_points,
        remove_points_after_sample=prep_cfg.remove_points_after_sample,
        remove_environment=prep_cfg.remove_environment,
        use_group_id=prep_cfg.use_group_id,
        out_size_factor=out_size_factor,       #8
        multi_gpu=multi_gpu,
        min_points_in_gt=prep_cfg.min_num_of_points_in_gt,
        random_flip_x=prep_cfg.random_flip_x,
        random_flip_y=prep_cfg.random_flip_y,
        sample_importance=prep_cfg.sample_importance)

    ret = target_assigner.generate_anchors(feature_map_size)
    class_names = target_assigner.classes   # ['car', 'bicycle', 'bus', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'barrier']
    anchors_dict = target_assigner.generate_anchors_dict(feature_map_size)
    anchors_list = []
    for k, v in anchors_dict.items():
        anchors_list.append(v["anchors"])  #50000个
    
    # anchors = ret["anchors"]
    anchors = np.concatenate(anchors_list, axis=0)
    anchors = anchors.reshape([-1, target_assigner.box_ndim])
    assert np.allclose(anchors, ret["anchors"].reshape(-1, target_assigner.box_ndim))
    matched_thresholds = ret["matched_thresholds"]
    unmatched_thresholds = ret["unmatched_thresholds"]
    anchors_bv = box_np_ops.rbbox2d_to_near_bbox(
        anchors[:, [0, 1, 3, 4, 6]])    #bev_anchor 4维
    anchor_cache = {
        "anchors": anchors,
        "anchors_bv": anchors_bv,
        "matched_thresholds": matched_thresholds,
        "unmatched_thresholds": unmatched_thresholds,
        "anchors_dict": anchors_dict,
    }
    prep_func = partial(prep_func, anchor_cache=anchor_cache)
    dataset = dataset_cls(
        info_path=dataset_cfg.kitti_info_path,        #数据的路径
        root_path=dataset_cfg.kitti_root_path,
        class_names=class_names,    #10个类
        prep_func=prep_func)

    return dataset           # _nusc_infos <class 'dict'>: {'lidar_path': '/home/lichao/v1.0-mini/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin', 'cam_front_path': '/home/lichao/v1.0-mini/samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg', 'token': 'ca9a282c9e77460f8360f564131a8af5', 'sweeps': [], 'lidar2ego_translation': [0.943713, 0.0, 1.84023], 'lidar2ego_rotation': [0.7077955119163518, -0.006492242056004365, 0.010646214713995808, -0.7063073142877817], 'ego2global_translation': [411.3039349319818, 1180.8903791765097, 0.0], 'ego2global_rotation': [0.5720320396729045, -0.0016977771610471074, 0.011798001930183783, -0.8201446642457809], 'timestamp': 1532402927647951, 'gt_boxes': array([[ 1.84143850e+01,  5.95160251e+01,  7.69634574e-01,
       #   6.21000000e-01,  6.69000000e-01,  1.64200000e+00,
       #  -4.69493230e+00],
       # [ 2.10021070e+01,  3.60611085e+01, -2.61477977e-02,
       #   7.75000000e-01,  7.69000000e-01,  1.71100000e+00,
       #  -3.09278986e+00],
       # [ 3.73518608e+01,  6.43973387e+01,  4.50991675e-01,
       #   2.01100000e+00,  4.63300000e+00,  1.57300000e+00,
       #  -4.65964174e+00],
       # [ 2.53650507e+01,  3.21656131e+01, -1.56578802e-01,
       #   7.52000000e-01,  8.19000000e-01,  1.63700000e+00,
       #   8.96434962e-02],
       # [ 6.63460939e+00, -1.53946443e+01, -1.81540646e+00,
       #   4.27000000e-01,  3.59000000e-01,  7.94000000e-01,
       #  -3.03749783e+00],
       # [ 1.85657828e+01,  6.08238271e+01,  6.84990021e-01,
       #   6.89000000e-01,  1.77000000e+00,  1.70900000e+00,
       #   1.42201039e+00],
       # [ 2.04209993e+01,  3.83205876e+01,  2.98603361e-02,
       #   6.61000000e-01,  7.03000000e-01,  1.83900000e+00,
       #  -3.07022275e+00],
       # [ 9.14824518e+00, -1.95423270e+01, -1.64500705e+00,
       #   1.83700000e+00,  4.32000000e+00,  1.63100000e+00,
       #   1.24270829e-01],
       # [-6.10498226e+00,  5.93584673e+01,  8.54154061e-01,
       #   6.48000000e-01,  7.09000000e-01,  1.60900000e+00,
       #  -1.87830250e+00],
       # [ 7.85689816e+00,  2.55566992e+01, -5.69874044e-01,
       #   1.97700000e+00,  7.03000000e-01,  1.14900000e+00,
       #  -4.64416067e+00],
       # [ 6.00786741e+00, -9.19556411e+00, -1.51171471e+00,
       #   1.91000000e+00,  5.55000000e-01,  1.05500000e+00,
       #  -4.65688412e+00],
       # [-1.35110040e+00, -1.49087806e+01, -1.33728763e+00,
       #   1.01900000e+00,  9.15000000e-01,  1.67000000e+00,
       #  -1.53396649e+00],
       # [-1.60726060e+01,  7.27176412e+00, -2.19296721e-01,
       #   9.34000000e-01,  8.91000000e-01,  1.83500000e+00,
       #  -3.19585155e+00],
       # [ 3.93926366e+01,  5.22074058e+01,  2.11055861e-01,
       #   8.00000000e-01,  9.08000000e-01,  1.83500000e+00,
       #  -4.69563043e+00],
       # [-2.17676621e+01, -4.58193819e-01, -4.12261939e-01,
       #   8.72000000e-01,  9.03000000e-01,  1.71900000e+00,
       #  -1.57807096e+00],
       # [ 7.96010949e+00,  2.75789489e+01, -4.96672223e-01,
       #   1.96400000e+00,  6.94000000e-01,  1.13100000e+00,
       #  -4.67934651e+00],
       # [ 5.97927366e+00,  3.50087253e+01,  4.40587610e-02,
       #   1.70800000e+00,  4.01000000e+00,  1.63100000e+00,
       #  -3.07271858e+00],
       # [ 1.93856351e+01,  6.12050230e+01,  6.55697594e-01,
       #   7.90000000e-01,  7.41000000e-01,  1.72500000e+00,
       #   1.51843984e+00],
       # [-4.49864330e+00,  1.52533225e+01,  3.96393503e-01,
       #   2.87700000e+00,  1.02010000e+01,  3.59500000e+00,
       #  -3.16598897e+00],
       # [-8.27170431e+00,  7.76698402e+01,  2.04984263e+00,
       #   2.13500000e+00,  4.95600000e+00,  2.17000000e+00,
       #  -1.51448862e+00],
       # [-2.26698534e+00,  6.33472982e+01,  1.43509478e+00,
       #   7.67000000e-01,  9.51000000e-01,  1.83500000e+00,
       #  -3.00561067e+00],
       # [ 9.53171813e+00,  4.22833332e+01,  1.08726265e-02,
       #   1.96700000e+00,  7.04000000e-01,  9.62000000e-01,
       #  -4.70147729e+00],
       # [ 8.13275639e+00,  3.15409624e+01, -3.81657453e-01,
       #   2.06000000e+00,  7.59000000e-01,  1.10000000e+00,
       #  -4.64443993e+00],
       # [ 8.63924568e+00,  1.93893837e+01, -8.56834835e-01,
       #   1.99300000e+00,  7.44000000e-01,  1.24500000e+00,
       #  -4.68262773e+00],
       # [ 6.89568022e+00,  9.48441421e+00, -1.12268335e+00,
       #   4.76000000e-01,  4.61000000e-01,  7.20000000e-01,
       #  -3.88832839e+00],
       # [ 7.09060535e+00,  1.55187191e+01, -8.49409976e-01,
       #   1.96900000e+00,  7.16000000e-01,  1.09200000e+00,
       #  -4.66829858e+00],
       # [ 8.02763054e+00, -5.38244203e+01, -1.48579682e+00,
       #   2.90900000e+00,  6.90800000e+00,  3.55800000e+00,
       #  -7.88550237e-03],
       # [-2.87553253e+01, -9.84972113e-01, -3.26194330e-01,
       #   8.03000000e-01,  8.70000000e-01,  1.78000000e+00,
       #  -4.70879022e+00],
       # [ 3.68071655e+01, -1.83079302e+01, -1.15671064e+00,
       #   8.42000000e-01,  8.84000000e-01,  1.74900000e+00,
       #  -4.02038000e+00],
       # [ 9.56223707e+00,  4.42745579e+01,  1.14128783e-01,
       #   1.99000000e+00,  7.26000000e-01,  9.21000000e-01,
       #  -4.67800261e+00],
       # [-4.26884008e+00,  1.30882430e+01,  9.89562424e-01,
       #   7.08000000e-01,  8.63000000e-01,  1.61600000e+00,
       #  -3.47449337e+00],
       # [ 2.01683999e+01,  3.58860512e+01, -5.44740228e-02,
       #   7.34000000e-01,  7.75000000e-01,  1.82100000e+00,
       #  -3.07079871e+00],
       # [ 8.54439373e+00,  1.73067688e+01, -8.90247602e-01,
       #   2.06900000e+00,  7.12000000e-01,  1.08800000e+00,
       #  -4.64772115e+00],
       # [ 2.07369050e+01,  4.21586392e+01,  1.06308444e-01,
       #   7.33000000e-01,  7.46000000e-01,  1.69500000e+00,
       #  -2.98381150e+00],
       # [-1.64776078e+00, -1.56464005e+01, -1.40857953e+00,
       #   9.13000000e-01,  8.73000000e-01,  1.69700000e+00,
       #  -1.49905991e+00],
       # [ 9.19537417e+00,  3.57739991e+01, -3.09876984e-01,
       #   2.05000000e+00,  6.93000000e-01,  1.08100000e+00,
       #  -4.69835315e+00],
       # [ 3.30117512e+00,  4.03404947e+01,  1.45804100e-01,
       #   1.84700000e+00,  4.11500000e+00,  1.52600000e+00,
       #  -3.07360869e+00],
       # [ 8.22300294e+00,  3.35988299e+01, -2.78988934e-01,
       #   1.97700000e+00,  7.64000000e-01,  1.14400000e+00,
       #  -4.69679980e+00],
       # [ 9.60979252e+00,  4.62887454e+01,  1.72893729e-01,
       #   1.84200000e+00,  6.77000000e-01,  8.85000000e-01,
       #  -4.69545590e+00],
       # [ 1.37566221e+01, -9.29551275e+00, -1.50533803e+00,
       #   7.93000000e-01,  1.00000000e+00,  1.60400000e+00,
       #   1.37424073e+00],
       # [ 2.95259452e+01,  6.50114578e+01,  5.75882031e-01,
       #   1.93900000e+00,  4.81900000e+00,  1.73600000e+00,
       #  -4.65964174e+00],
       # [ 6.98584210e+00,  1.14208824e+01, -9.44204535e-01,
       #   2.07300000e+00,  6.33000000e-01,  1.07800000e+00,
       #  -4.70800482e+00],
       # [ 7.72508647e+00,  2.35633192e+01, -6.14605547e-01,
       #   1.94500000e+00,  6.49000000e-01,  1.11000000e+00,
       #  -4.62670738e+00],
       # [-1.23350597e+01,  6.99007654e+01,  2.59496180e+00,
       #   3.01600000e+00,  3.99200000e+00,  2.91600000e+00,
       #  -1.51448862e+00],
       # [ 7.19791594e+00,  1.75240138e+01, -7.67858142e-01,
       #   1.96400000e+00,  7.20000000e-01,  1.10400000e+00,
       #  -4.63861053e+00],
       # [ 3.78552674e+01,  7.09529966e+01,  6.90733989e-01,
       #   1.97200000e+00,  4.69800000e+00,  1.58100000e+00,
       #  -4.68355276e+00],
       # [-6.15260058e-01,  6.18880681e+01,  1.16346064e+00,
       #   7.59000000e-01,  8.37000000e-01,  1.91600000e+00,
       #  -3.02306396e+00],
       # [ 2.16574469e+01,  4.19727942e+01, -8.84870833e-02,
       #   7.39000000e-01,  7.71000000e-01,  1.76600000e+00,
       #  -2.99304430e+00],
       # [-1.88415861e+00,  6.73835906e+01,  1.51044737e+00,
       #   6.94000000e-01,  9.02000000e-01,  1.80700000e+00,
       #  -2.99606371e+00],
       # [ 5.90503391e+00, -1.03553804e+01, -1.64179728e+00,
       #   3.36000000e-01,  3.32000000e-01,  6.93000000e-01,
       #  -4.63719681e+00],
       # [ 2.07435172e+01,  3.76223744e+01, -4.95785876e-02,
       #   7.51000000e-01,  8.07000000e-01,  1.70600000e+00,
       #  -3.01786288e+00],
       # [ 1.34211510e+00,  6.04788048e+01,  9.72934321e-01,
       #   7.77000000e-01,  9.58000000e-01,  1.75700000e+00,
       #  -1.54209973e+00],
       # [ 6.70496413e+00,  4.57677899e+01,  6.48678724e-01,
       #   1.78700000e+00,  4.53500000e+00,  2.05900000e+00,
       #  -3.05615540e+00],
       # [-3.84302626e+00, -1.36188136e+01, -1.13770914e+00,
       #   9.42000000e-01,  1.04000000e+00,  1.93700000e+00,
       #  -1.62123296e+00],
       # [ 2.10460456e+01,  6.22610968e+01,  7.68191868e-01,
       #   6.87000000e-01,  7.16000000e-01,  1.73000000e+00,
       #  -3.22332304e+00],
       # [ 3.45913764e+01, -3.18015561e+01, -1.51789431e+00,
       #   9.00000000e-01,  9.56000000e-01,  1.82700000e+00,
       #  -1.99717902e+00],
       # [-1.50833275e+00,  6.31356336e+01,  1.28291617e+00,
       #   7.28000000e-01,  8.58000000e-01,  2.00000000e+00,
       #  -3.07542384e+00],
       # [ 3.08152535e+01, -1.12317707e+01, -1.08402278e+00,
       #   7.67000000e-01,  8.72000000e-01,  1.80900000e+00,
       #  -1.02108404e+00],
       # [-2.51817183e+00,  1.68564583e+01, -4.72614255e-01,
       #   6.34000000e-01,  6.18000000e-01,  1.75200000e+00,
       #   1.26685062e+00],
       # [-2.80841235e+00,  1.67430847e+01, -6.90276569e-01,
       #   5.99000000e-01,  8.35000000e-01,  1.26500000e+00,
       #  -3.17405239e+00],
       # [ 6.62182278e+00, -9.23805105e+00, -1.54466405e+00,
       #   1.90800000e+00,  5.79000000e-01,  1.05100000e+00,
       #  -4.65103727e+00],
       # [ 9.13430371e+00,  3.37413951e+01, -3.78085223e-01,
       #   2.04200000e+00,  7.14000000e-01,  1.07300000e+00,
       #  -4.68089985e+00],
       # [-1.81516999e+00, -1.35683945e+01, -1.31625287e+00,
       #   9.71000000e-01,  9.37000000e-01,  1.56800000e+00,
       #  -1.63868625e+00],
       # [ 8.22821175e+00,  1.16164901e+01, -9.92498348e-01,
       #   2.12600000e+00,  7.16000000e-01,  1.03100000e+00,
       #   1.54027390e+00],
       # [ 7.34853252e+00,  1.95358624e+01, -7.19767294e-01,
       #   2.03700000e+00,  7.31000000e-01,  1.10000000e+00,
       #  -4.63107070e+00],
       # [-2.05323576e+00,  3.80260869e+01,  2.70261055e-01,
       #   1.90700000e+00,  4.72700000e+00,  1.95700000e+00,
       #  -3.15125839e+00],
       # [ 7.51887413e+00,  2.15547621e+01, -6.93749190e-01,
       #   2.03400000e+00,  6.83000000e-01,  1.12700000e+00,
       #  -4.61750950e+00],
       # [ 8.03831388e+00,  2.95530758e+01, -4.34777454e-01,
       #   1.90800000e+00,  7.21000000e-01,  1.10600000e+00,
       #  -4.66189322e+00],
       # [ 7.03563082e+00,  1.34548224e+01, -9.31816877e-01,
       #   1.99000000e+00,  6.51000000e-01,  1.10700000e+00,
       #  -4.70221032e+00]]), 'gt_names': array(['pedestrian', 'pedestrian', 'car', 'pedestrian', 'traffic_cone',
       # 'bicycle', 'pedestrian', 'car', 'pedestrian', 'barrier', 'barrier',
       # 'pedestrian', 'pedestrian', 'pedestrian', 'pedestrian', 'barrier',
       # 'car', 'pedestrian', 'truck', 'car', 'pedestrian', 'barrier',
       # 'barrier', 'barrier', 'traffic_cone', 'barrier', 'bus',
       # 'pedestrian', 'pedestrian', 'barrier', 'pedestrian', 'pedestrian',
       # 'barrier', 'pedestrian', 'pedestrian', 'barrier', 'car', 'barrier',
       # 'barrier', 'pedestrian', 'car', 'barrier', 'barrier',
       # 'construction_vehicle', 'barrier', 'car', 'pedestrian',
       # 'pedestrian', 'pedestrian', 'traffic_cone', 'pedestrian',
       # 'pedestrian', 'truck', 'pedestrian', 'pedestrian', 'pedestrian',
       # 'pedestrian', 'pedestrian', 'pedestrian',
       # 'movable_object.pushable_pullable', 'barrier', 'barrier',
       # 'pedestrian', 'barrier', 'barrier', 'car', 'barrier', 'barrier',
       # 'barrier'], dtype='<U32'), 'gt_velocity': array([[ 0.00000000e+00,  0.00000000e+00],
       # [ 3.57412927e-02,  1.25839028e+00],
       # [ 3.93090486e-02, -2.63784664e-03],
       # [-5.32093020e-02, -1.42018271e+00],
       # [-2.33876851e-03, -2.85297124e-02],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 1.36054920e-01,  9.84323482e-01],
       # [-7.40969783e-01, -9.53975822e+00],
       # [ 1.32730129e+00,  5.34916646e-01],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 1.13511559e+00,  4.73227328e-02],
       # [-1.50420027e-01,  1.65642322e+00],
       # [-1.28953923e+00,  1.94160778e-01],
       # [            nan,             nan],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 2.60680039e-01,  1.68559917e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [-2.72124004e-02,  2.19677919e-02],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 1.16048341e-01,  1.17284138e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [-1.65339182e-02,  6.84519482e-02],
       # [-8.20094887e-03, -8.73832868e-04],
       # [ 1.65602380e-01, -9.72937882e+00],
       # [            nan,             nan],
       # [-1.11103088e+00,  9.57738892e-01],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 1.21893492e-01,  1.51770627e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 1.79872223e-01,  1.21962477e+00],
       # [ 1.35732305e+00,  1.09519533e-01],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 5.69111522e-01,  1.12372931e+01],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [-2.68190654e+00, -4.53984537e-01],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 2.41306922e-01, -7.44456749e-03],
       # [ 2.49212975e-01,  1.51323148e+00],
       # [ 3.94914830e-02,  1.17248556e+00],
       # [ 3.38533328e-01,  1.26919940e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 9.81775264e-02,  1.13436757e+00],
       # [-2.34114589e-02, -5.65780493e-05],
       # [ 3.10095144e-01,  3.21508648e+00],
       # [ 9.16766570e-01,  1.14262141e-01],
       # [-2.83597812e-02, -3.99699997e-03],
       # [ 8.11775406e-01,  3.31269518e-01],
       # [ 2.25920048e-01,  1.26841484e+00],
       # [ 1.11389419e+00, -6.62926414e-01],
       # [ 1.66286762e-02, -2.79714659e-02],
       # [-1.91196981e-02, -1.92199307e-01],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [ 1.33808482e+00,  1.62080289e-01],
       # [ 2.26778434e-04, -2.97191316e-02],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [-4.92173273e-02,  5.18032866e+00],
       # [ 0.00000000e+00,  0.00000000e+00],
       # [-6.87079640e-04,  1.87716987e-03],
       # [ 0.00000000e+00,  0.00000000e+00]]), 'num_lidar_pts': array([  1,   2,   5,   1,   1,   1,   1,  45,   1,   4,  77,   7,   6,
       #   1,   8,   2,   4,   1, 495,   1,   1,   3,   3,   2,   8,  19,
       #   3,   5,   3,   1,   0,   2,   5,   3,  14,   2,   5,   5,   1,
       #   4,   2,  50,   4,   4,  13,   2,   0,   2,   1,   4,   1,   0,
       #   7,  12,   1,   2,   1,   5,  13,  10,  20,   1,  10,  32,   9,
       #  15,   6,   2,  27]), 'num_radar_pts': array([ 0,  0,  0,  0,  0,  0,  0,  6,  0,  2,  0,  3,  0,  0,  0,  1,  2,
       #  0, 13,  0,  0,  2,  1,  1,  0,  2,  2,  0,  0,  0,  0,  0,  0,  0,
       #  1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
       #  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  1,
       #  2])}
