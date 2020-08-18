import json
import pickle
import time
import random
from copy import deepcopy
from functools import partial
from pathlib import Path
import subprocess
from pyquaternion import Quaternion
from typing import Tuple, List, Dict
import fire
import numpy as np
import struct

from second.core import box_np_ops
from second.core import preprocess as prep
from second.data import kitti_common as kitti
from second.data.dataset import Dataset, register_dataset
from second.utils.eval import get_coco_eval_result, get_official_eval_result
from second.utils.progress_bar import progress_bar_iter as prog_bar
from second.utils.timer import simple_timer


@register_dataset
class NuScenesDataset(Dataset):
    NumPointFeatures = 4  # xyz, timestamp. set 4 to use kitti pretrain
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.parked",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }

    def __init__(self,
                 root_path,
                 info_path,
                 class_names=None,
                 prep_func=None,
                 num_point_features=None):
        self._root_path = Path(root_path)
        with open(info_path, 'rb') as f:
            data = pickle.load(f)
        self._nusc_infos = data["infos"]
        self._nusc_infos = list(
            sorted(self._nusc_infos, key=lambda e: e["timestamp"]))   #根据info信息加载lidar和camera的每张图路径，每张图片的label,lidartoego
        self._metadata = data["metadata"]
        self._class_names = class_names  #<class 'list'>: ['car', 'bicycle', 'bus', 'construction_vehicle', 'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck', 'barrier']
        self._prep_func = prep_func
        # kitti map: nusc det name -> kitti eval name
        self._kitti_name_mapping = {
            "car": "car",
            "pedestrian": "pedestrian",
        }  # we only eval these classes in kitti
        self.version = self._metadata["version"]
        self.eval_version = "cvpr_2019"
        self._with_velocity = False

    def __len__(self):
        return len(self._nusc_infos)

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self._nusc_infos[0]:
            return None
        from second.data.nuscenes.eval.detection.config import eval_detection_configs
        cls_range_map = eval_detection_configs[self.
                                               eval_version]["class_range"]
        gt_annos = []
        for info in self._nusc_infos:
            gt_names = info["gt_names"]
            gt_boxes = info["gt_boxes"]
            num_lidar_pts = info["num_lidar_pts"]
            mask = num_lidar_pts > 0
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            num_lidar_pts = num_lidar_pts[mask]

            mask = np.array([n in self._kitti_name_mapping for n in gt_names],
                            dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            num_lidar_pts = num_lidar_pts[mask]
            gt_names_mapped = [self._kitti_name_mapping[n] for n in gt_names]
            det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            num_lidar_pts = num_lidar_pts[mask]
            # use occluded to control easy/moderate/hard in kitti
            easy_mask = num_lidar_pts > 15
            moderate_mask = num_lidar_pts > 7
            occluded = np.zeros([num_lidar_pts.shape[0]])
            occluded[:] = 2
            occluded[moderate_mask] = 1
            occluded[easy_mask] = 0
            N = len(gt_boxes)
            gt_annos.append({
                "bbox":
                np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                "alpha":
                np.full(N, -10),
                "occluded":
                occluded,
                "truncated":
                np.zeros(N),
                "name":
                gt_names,
                "location":
                gt_boxes[:, :3],
                "dimensions":
                gt_boxes[:, 3:6],
                "rotation_y":
                gt_boxes[:, 6],
            })
        return gt_annos

    def __getitem__(self, idx):   #37
        input_dict = self.get_sensor_data(idx)
        example = self._prep_func(input_dict=input_dict)
        example["metadata"] = input_dict["metadata"]
        if "anchors_mask" in example:
            example["anchors_mask"] = example["anchors_mask"].astype(np.uint8)
        return example

    def get_sensor_data(self, query):
        idx = query
        read_test_image = False
        if isinstance(query, dict):         #判断一个对象是否是一个已知的类型
            assert "lidar" in query
            idx = query["lidar"]["idx"]
            read_test_image = "cam" in query

        info = self._nusc_infos[idx]
        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
            },
            "radar":{
                "type":"radar",
                "radar_points": None ,
            },
            "metadata": {
                "token": info["token"]
            },
        }
        lidar_path = Path(info['lidar_path'])  #/home/lichao/v1.0-mini/samples/LIDAR_TOP/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801001947564.pcd.bin
        points = np.fromfile(
            str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
        points[:, 3] /= 255
        points[:, 4] = 0
        sweep_points_list = [points]
        ts = info["timestamp"] / 1e6

        for sweep in info["sweeps"]:
            points_sweep = np.fromfile(
                str(sweep["lidar_path"]), dtype=np.float32,
                count=-1).reshape([-1, 5])
            sweep_ts = sweep["timestamp"] / 1e6
            points_sweep[:, 3] /= 255
            # aa=print(points_sweep[:, :3])
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[   #@表示向量乘法
                "sweep2lidar_rotation"].T
            #print(points_sweep[:, :3])
            points_sweep[:, :3] += sweep["sweep2lidar_translation"]
            points_sweep[:, 4] = ts - sweep_ts
            sweep_points_list.append(points_sweep)


        points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]  #把反射强度给去掉了
        all_radar_points=[]
        radar_name=['radar_front_path','radar_front_right_path','radar_front_left_path','radar_back_left_path','radar_back_right_path']
        radar_rotation_name=['radar_front2lidar_rotation','radar_front_right2lidar_rotation','radar_front_left2lidar_rotation','radar_back_left2lidar_rotation','radar_back_left2lidar_rotation']
        radar_translation_name=['radar_front2lidar_translation','radar_front_right2lidar_translation','radar_front_left2lidar_translation','radar_back_left2lidar_translation','radar_back_right2lidar_translation']
        for i in range(5):
            radar_path = Path(info[str(radar_name[i])])
            radar_points = from_file(str(radar_path),invalid_states=[0],
                                       dynprop_states=list(range(7)),
                                       ambig_states=[3])
            if radar_points.size!=0:
                radar_points[:, :3] = radar_points[:,:3] @ info[str(radar_rotation_name[i])]
                radar_points[:, :3] -= info[str(radar_translation_name[i])]
                radar_points_velocities=radar_points[:,8:10]
                radar_points_velocities=np.hstack((radar_points_velocities,np.zeros([radar_points.shape[0],1])))
                radar_points_velocities=radar_points_velocities @ info[str(radar_rotation_name[i])]
                radar_points[:,8:10] = radar_points_velocities[:,0:2]
            else:
                radar_points=np.zeros([1,18])
            radar_points=radar_points.reshape([-1,18])
            radar_sweeps_points_list = [radar_points]
            for radar_sweeps in info["radar_sweeps"]:
                radar_points_sweeps= from_file(str(radar_sweeps[str(radar_name[i])]), invalid_states=[0],
                                     dynprop_states=list(range(7)),
                                     ambig_states=[3])
                if radar_points_sweeps.size == 0:
                    radar_points_sweeps = np.zeros([1, 18])
                radar_points_sweeps=radar_points_sweeps.reshape([-1, 18])
                radar_points_sweeps[:, :3] = radar_points_sweeps[:, :3] @ radar_sweeps[
                    str(radar_rotation_name[i])]
                radar_points_sweeps[:, :3] -= radar_sweeps[str(radar_translation_name[i])]
                radar_sweeps_points_velocities = radar_points_sweeps[:, 8:10]
                radar_sweeps_points_velocities = np.hstack((radar_sweeps_points_velocities, np.zeros([radar_points_sweeps.shape[0], 1])))
                radar_sweeps_points_velocities = radar_sweeps_points_velocities @ info[str(radar_rotation_name[i])]
                radar_points_sweeps[:, 8:10] = radar_sweeps_points_velocities[:, 0:2]
                radar_sweeps_points_list.append(radar_points_sweeps)
            try:
                radar_points = np.concatenate(radar_sweeps_points_list, axis=0)[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
            except:
                print(radar_sweeps_points_list.shape)
            # radar_points.append(radar_points)     #3150个点还多了？
            all_radar_points.append(radar_points)
        all_radar_points=np.concatenate(all_radar_points,axis=0)[:,[0, 1, 2, 5, 9, 10]]
        # print(all_radar_points.shape)



################################radar_front
        # radar_front_path=Path(info['radar_front_path'])
        # radar_front_points = from_file(str(radar_front_path), invalid_states=list(range(17 + 1)),
        #                                dynprop_states=list(range(7 + 1)),
        #                                ambig_states=list(range(4 + 1)))
#         radar_front_points[:, :3] = radar_front_points[:, :3] @ info["radar_front2lidar_rotation"]
#         radar_front_points[:, :3] -= info["radar_front2lidar_translation"]
#         sweeps_radar_front_points_list= [radar_front_points]
#         radar_front_right_empty_num = 0
#         radar_front_left_empty_num = 0
#         radar_back_left_empty_num = 0
#         radar_back_right_empty_num = 0
#         for radar_sweeps in info["radar_sweeps"]:
#             radar_front_points_sweeps = from_file(str(radar_sweeps["radar_front_path"]), invalid_states=list(range(17 + 1)),
#                                      dynprop_states=list(range(7 + 1)),
#                                      ambig_states=list(range(4 + 1)))
#             a=radar_front_points_sweeps[:, :3]
#             radar_front_points_sweeps[:, :3] = radar_front_points_sweeps[:, :3] @ radar_sweeps["radar_front2lidar_rotation"]
#             radar_front_points_sweeps[:, :3] -= radar_sweeps["radar_front2lidar_translation"]
#             b=radar_front_points_sweeps[:, :3]
#             sweeps_radar_front_points_list.append(radar_front_points_sweeps)
#         radar_front_points = np.concatenate(sweeps_radar_front_points_list, axis=0)[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]]
# ###########################################################################
#
#         radar_front_right_path = Path(info['radar_front_right_path'])
#         radar_front_right_points = from_file(str(radar_front_right_path), invalid_states=list(range(17 + 1)),
#                                        dynprop_states=list(range(7 + 1)),
#                                        ambig_states=list(range(4 + 1)))
#         radar_front_right_points = radar_front_right_points.reshape([-1, 18])
#         radar_front_right_points[:, :3] = radar_front_right_points[:, :3] @ info["radar_front_right2lidar_rotation"]
#         radar_front_right_points[:, :3] -= info["radar_front_right2lidar_translation"]
#         sweeps_radar_front_right_points_list = [radar_front_right_points]
#         for radar_sweeps in info["radar_sweeps"]:
#             radar_front_right_points_sweeps = from_file(str(radar_sweeps["radar_front_right_path"]),
#                                                   invalid_states=list(range(17 + 1)),
#                                                   dynprop_states=list(range(7 + 1)),
#                                                   ambig_states=list(range(4 + 1)))
#             # if radar_sweeps["radar_front_right2lidar_rotation"].shape(0)==3:
#             # try:
#             #     radar_front_right_points_sweeps[:, :3] = radar_front_right_points_sweeps[:, :3] @ radar_sweeps["radar_front_right2lidar_rotation"]
#             # except:
#             #     print("error",radar_sweeps["radar_front_right2lidar_rotation"])
#             #     print(radar_sweeps["radar_front_right_path"])
#             #     print("shape",radar_front_right_points_sweeps.shape)
#             if radar_front_right_points_sweeps.size==0:
#                 one = np.ones([1, 18])
#                 radar_front_right_points_sweeps=one
#                 radar_front_right_empty_num=radar_front_right_empty_num+1
#                 print("radar_front_right_empty_num",radar_front_right_empty_num)
#             radar_front_right_points_sweeps[:, :3] = radar_front_right_points_sweeps[:, :3] @ radar_sweeps[
#                     "radar_front_right2lidar_rotation"]
#             radar_front_right_points_sweeps[:, :3] -= radar_sweeps["radar_front_right2lidar_translation"]
#
#             sweeps_radar_front_right_points_list.append(radar_front_right_points_sweeps)
#         radar_front_right_points  = np.concatenate(sweeps_radar_front_right_points_list, axis=0)[:,
#                              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
# #############################################################
#         radar_front_left_path = Path(info['radar_front_left_path'])
#         radar_front_left_points = from_file(str(radar_front_left_path), invalid_states=list(range(17 + 1)),
#                                              dynprop_states=list(range(7 + 1)),
#                                              ambig_states=list(range(4 + 1)))
#         radar_front_left_points = radar_front_left_points.reshape([-1,18])
#         radar_front_left_points[:, :3] = radar_front_left_points[:, :3] @ info["radar_front_left2lidar_rotation"]
#         radar_front_left_points[:, :3] -= info["radar_front_left2lidar_translation"]
#         sweeps_radar_front_left_points_list = [radar_front_left_points]
#         for radar_sweeps in info["radar_sweeps"]:
#             radar_front_left_points_sweeps = from_file(str(radar_sweeps["radar_front_left_path"]),
#                                                         invalid_states=list(range(17 + 1)),
#                                                         dynprop_states=list(range(7 + 1)),
#                                                         ambig_states=list(range(4 + 1)))
#             radar_front_left_points_sweeps = radar_front_left_points_sweeps.reshape([-1,18])
#             # print(str(radar_sweeps["radar_front_left_path"]))
#             a = radar_front_left_points_sweeps[:, :3]
#             if radar_front_left_points_sweeps.shape[1]==0:
#                 one = np.ones([1, 18])
#                 radar_front_left_points_sweeps=one
#                 radar_front_left_empty_num=radar_front_left_empty_num+1
#                 print("radar_front_left_empty_num",radar_front_left_empty_num)
#             radar_front_left_points_sweeps[:, :3] = radar_front_left_points_sweeps[:, :3] @ radar_sweeps[
#                 "radar_front_left2lidar_rotation"]
#             radar_front_left_points_sweeps[:, :3] -= radar_sweeps["radar_front_left2lidar_translation"]
#             b = radar_front_left_points_sweeps[:, :3]
#             sweeps_radar_front_left_points_list.append(radar_front_left_points_sweeps)
#         radar_front_left_points = np.concatenate(sweeps_radar_front_left_points_list, axis=0)[:,
#                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
# #################################################################################
#         radar_back_right_path = Path(info['radar_back_right_path'])
#         radar_back_right_points = from_file(str(radar_back_right_path), invalid_states=list(range(17 + 1)),
#                                              dynprop_states=list(range(7 + 1)),
#                                              ambig_states=list(range(4 + 1)))
#         radar_back_right_points[:, :3] = radar_back_right_points[:, :3] @ info["radar_back_right2lidar_rotation"]
#         radar_back_right_points[:, :3] -= info["radar_back_right2lidar_translation"]
#         sweeps_radar_back_right_points_list = [radar_back_right_points]
#         for radar_sweeps in info["radar_sweeps"]:
#             radar_back_right_points_sweeps = from_file(str(radar_sweeps["radar_back_right_path"]),
#                                                         invalid_states=list(range(17 + 1)),
#                                                         dynprop_states=list(range(7 + 1)),
#                                                         ambig_states=list(range(4 + 1)))
#             radar_back_right_points_sweeps = radar_back_right_points_sweeps.reshape([-1, 18])
#             a = radar_back_right_points_sweeps[:, :3]
#             if radar_back_right_points_sweeps.shape[1]==0:
#                 one = np.ones([1, 18])
#                 radar_back_right_points_sweeps=one
#                 radar_back_right_empty_num=radar_back_right_empty_num+1
#                 print("radar_back_right_empty_num",radar_back_right_empty_num)
#             radar_back_right_points_sweeps[:, :3] = radar_back_right_points_sweeps[:, :3] @ radar_sweeps[
#                 "radar_back_right2lidar_rotation"]
#             radar_back_right_points_sweeps[:, :3] -= radar_sweeps["radar_back_right2lidar_translation"]
#             b = radar_back_right_points_sweeps[:, :3]
#             sweeps_radar_back_right_points_list.append(radar_back_right_points_sweeps)
#         radar_back_right_points = np.concatenate(sweeps_radar_back_right_points_list, axis=0)[:,
#                                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
#
#         radar_back_left_path = Path(info['radar_back_left_path'])
#         radar_back_left_points = from_file(str(radar_back_left_path), invalid_states=list(range(17 + 1)),
#                                             dynprop_states=list(range(7 + 1)),
#                                             ambig_states=list(range(4 + 1)))
#         radar_back_left_points[:, :3] = radar_back_left_points[:, :3] @ info["radar_back_left2lidar_rotation"]
#         radar_back_left_points[:, :3] -= info["radar_back_left2lidar_translation"]
#         sweeps_radar_back_left_points_list = [radar_back_left_points]
#         for radar_sweeps in info["radar_sweeps"]:
#             radar_back_left_points_sweeps = from_file(str(radar_sweeps["radar_back_left_path"]),
#                                                        invalid_states=list(range(17 + 1)),
#                                                        dynprop_states=list(range(7 + 1)),
#                                                        ambig_states=list(range(4 + 1)))
#             radar_back_left_points_sweeps = radar_back_left_points_sweeps.reshape([-1, 18])
#             a = radar_back_left_points_sweeps[:, :3]
#             if radar_back_left_points_sweeps.shape[1]==0:
#                 one = np.ones([1, 18])
#                 radar_back_left_points_sweeps=one
#                 radar_back_left_empty_num=radar_back_left_empty_num+1
#                 print("radar_back_left_empty_num",radar_back_left_empty_num)
#             radar_back_left_points_sweeps[:, :3] = radar_back_left_points_sweeps[:, :3] @ radar_sweeps[
#                 "radar_back_left2lidar_rotation"]
#             radar_back_left_points_sweeps[:, :3] -= radar_sweeps["radar_back_left2lidar_translation"]
#             b = radar_back_left_points_sweeps[:, :3]
#             sweeps_radar_back_left_points_list.append(radar_back_left_points_sweeps)
#         radar_back_left_points = np.concatenate(sweeps_radar_back_left_points_list, axis=0)[:,
#                                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
#
#         all_radar_points=np.concatenate((radar_front_points, radar_front_left_points, radar_front_right_points,radar_back_left_points,radar_back_right_points),axis=0)[:,      #3150个点还多了？
#                                   [0, 1, 2, 5]]
        # print(all_radar_points.dtype)
        all_radar_points.astype(np.float32)

        if read_test_image:
            if Path(info["cam_front_path"]).exists():
                with open(str(info["cam_front_path"]), 'rb') as f:
                    image_str = f.read()
            else:
                image_str = None
            res["cam"] = {
                "type": "camera",
                "data": image_str,
                "datatype": Path(info["cam_front_path"]).suffix[1:],
            }
        res["lidar"]["points"] = points
        res["radar"]["radar_points"] = all_radar_points
        if 'gt_boxes' in info:
            mask = info["num_lidar_pts"] > 0
            gt_boxes = info["gt_boxes"][mask]
            if self._with_velocity:
                gt_velocity = info["gt_velocity"][mask]
                nan_mask = np.isnan(gt_velocity[:, 0])
                gt_velocity[nan_mask] = [0.0, 0.0]
                gt_boxes = np.concatenate([gt_boxes, gt_velocity], axis=-1)
            res["lidar"]["annotations"] = {
                'boxes': gt_boxes,
                'names': info["gt_names"][mask],
            }
        return res

    def evaluation_kitti(self, detections, output_dir):
        """eval by kitti evaluation tool.
        I use num_lidar_pts to set easy, mod, hard.
        easy: num>15, mod: num>7, hard: num>0.
        """
        print("++++++++NuScenes KITTI unofficial Evaluation:")
        print(
            "++++++++easy: num_lidar_pts>15, mod: num_lidar_pts>7, hard: num_lidar_pts>0"
        )
        print("++++++++The bbox AP is invalid. Don't forget to ignore it.")
        class_names = self._class_names
        gt_annos = self.ground_truth_annotations
        if gt_annos is None:
            return None
        gt_annos = deepcopy(gt_annos)
        detections = deepcopy(detections)
        dt_annos = []
        for det in detections:
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()
            anno = kitti.get_start_result_anno()
            num_example = 0
            box3d_lidar = final_box_preds
            for j in range(box3d_lidar.shape[0]):
                anno["bbox"].append(np.array([0, 0, 50, 50]))
                anno["alpha"].append(-10)
                anno["dimensions"].append(box3d_lidar[j, 3:6])
                anno["location"].append(box3d_lidar[j, :3])
                anno["rotation_y"].append(box3d_lidar[j, 6])
                anno["name"].append(class_names[int(label_preds[j])])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores[j])
                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                dt_annos.append(anno)
            else:
                dt_annos.append(kitti.empty_result_anno())
            num_example = dt_annos[-1]["name"].shape[0]
            dt_annos[-1]["metadata"] = det["metadata"]

        for anno in gt_annos:
            names = anno["name"].tolist()
            mapped_names = []
            for n in names:
                if n in self.NameMapping:
                    mapped_names.append(self.NameMapping[n])
                else:
                    mapped_names.append(n)
            anno["name"] = np.array(mapped_names)
        for anno in dt_annos:
            names = anno["name"].tolist()
            mapped_names = []
            for n in names:
                if n in self.NameMapping:
                    mapped_names.append(self.NameMapping[n])
                else:
                    mapped_names.append(n)
            anno["name"] = np.array(mapped_names)
        mapped_class_names = []
        for n in self._class_names:
            if n in self.NameMapping:
                mapped_class_names.append(self.NameMapping[n])
            else:
                mapped_class_names.append(n)

        z_axis = 2
        z_center = 0.5
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.
        result_official_dict = get_official_eval_result(
            gt_annos,
            dt_annos,
            mapped_class_names,
            z_axis=z_axis,
            z_center=z_center)
        result_coco = get_coco_eval_result(
            gt_annos,
            dt_annos,
            mapped_class_names,
            z_axis=z_axis,
            z_center=z_center)
        return {
            "results": {
                "official": result_official_dict["result"],
                "coco": result_coco["result"],
            },
            "detail": {
                "official": result_official_dict["detail"],
                "coco": result_coco["detail"],
            },
        }

    def evaluation_nusc(self, detections, output_dir):
        version = self.version
        eval_set_map = {
            # "v1.0-mini": "mini_train",
            "v1.0-mini":"mini_val",
            "v1.0-trainval": "val",
        }
        gt_annos = self.ground_truth_annotations
        if gt_annos is None:
            return None
        nusc_annos = {}
        mapped_class_names = self._class_names
        token2info = {}
        for info in self._nusc_infos:
            token2info[info["token"]] = info
        for det in detections:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                velocity = box.velocity[:2].tolist()
                if len(token2info[det["metadata"]["token"]]["sweeps"]) == 0:
                    velocity = (np.nan, np.nan)
                box.velocity = np.array([*velocity, 0.0])
            boxes = _lidar_nusc_box_to_global(
                token2info[det["metadata"]["token"]], boxes,
                mapped_class_names, "cvpr_2019")
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                velocity = box.velocity[:2].tolist()
                nusc_anno = {
                    "sample_token": det["metadata"]["token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": velocity,
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": NuScenesDataset.DefaultAttribute[name],
                }
                annos.append(nusc_anno)
            nusc_annos[det["metadata"]["token"]] = annos
        nusc_submissions = {
            "meta": {
                "use_camera": False,
                "use_lidar": False,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "results": nusc_annos,
        }
        res_path = Path(output_dir) / "results_nusc.json"
        with open(res_path, "w") as f:
            json.dump(nusc_submissions, f)
        eval_main_file = Path(__file__).resolve().parent / "nusc_eval.py"
        # why add \"{}\"? to support path with spaces.
        cmd = f"python {str(eval_main_file)} --root_path=\"{str(self._root_path)}\""
        cmd += f" --version={self.version} --eval_version={self.eval_version}"
        cmd += f" --res_path=\"{str(res_path)}\" --eval_set={eval_set_map[self.version]}"
        cmd += f" --output_dir=\"{output_dir}\""
        # use subprocess can release all nusc memory after evaluation
        subprocess.check_output(cmd, shell=True)
        with open(Path(output_dir) / "metrics_summary.json", "r") as f:
            metrics = json.load(f)
        detail = {}
        res_path.unlink()  # delete results_nusc.json since it's very large
        result = f"Nusc {version} Evaluation\n"
        for name in mapped_class_names:
            detail[name] = {}
            for k, v in metrics["label_aps"][name].items():
                detail[name][f"dist@{k}"] = v
            tp_errs = []
            tp_names = []
            for k, v in metrics["label_tp_errors"][name].items():
                detail[name][k] = v
                tp_errs.append(f"{v:.4f}")
                tp_names.append(k)
            threshs = ', '.join(list(metrics["label_aps"][name].keys()))
            scores = list(metrics["label_aps"][name].values())
            scores = ', '.join([f"{s * 100:.2f}" for s in scores])
            result += f"{name} Nusc dist AP@{threshs} and TP errors\n"
            result += scores
            result += "\n"
            result += ', '.join(tp_names) + ": " + ', '.join(tp_errs)
            result += "\n"
        return {
            "results": {
                "nusc": result
            },
            "detail": {
                "nusc": detail
            },
        }

    def evaluation(self, detections, output_dir):
        """kitti evaluation is very slow, remove it.
        """
        # res_kitti = self.evaluation_kitti(detections, output_dir)
        res_nusc = self.evaluation_nusc(detections, output_dir)
        res = {
            "results": {
                "nusc": res_nusc["results"]["nusc"],
                # "kitti.official": res_kitti["results"]["official"],
                # "kitti.coco": res_kitti["results"]["coco"],
            },
            "detail": {
                "eval.nusc": res_nusc["detail"]["nusc"],
                # "eval.kitti": {
                #     "official": res_kitti["detail"]["official"],
                #     "coco": res_kitti["detail"]["coco"],
                # },
            },
        }
        return res


@register_dataset
class NuScenesDatasetD8(NuScenesDataset):
    """Nuscenes mini train set. only contains ~3500 samples.
    recommend to use this to develop, train full set once before submit.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if len(self._nusc_infos) > 28000:
            self._nusc_infos = list(
                sorted(self._nusc_infos, key=lambda e: e["timestamp"]))
            self._nusc_infos = self._nusc_infos[::8]


@register_dataset
class NuScenesDatasetD8Velo(NuScenesDatasetD8):
    """Nuscenes mini train set with velocity.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._with_velocity = True


@register_dataset
class NuScenesDatasetVelo(NuScenesDataset):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._with_velocity = True


@register_dataset
class NuScenesDatasetD7(NuScenesDataset):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if len(self._nusc_infos) > 28000:
            self._nusc_infos = list(
                sorted(self._nusc_infos, key=lambda e: e["timestamp"]))
            self._nusc_infos = self._nusc_infos[::7]


@register_dataset
class NuScenesDatasetD6(NuScenesDataset):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if len(self._nusc_infos) > 28000:
            self._nusc_infos = list(
                sorted(self._nusc_infos, key=lambda e: e["timestamp"]))
            self._nusc_infos = self._nusc_infos[::6]


@register_dataset
class NuScenesDatasetD5(NuScenesDataset):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if len(self._nusc_infos) > 28000:
            self._nusc_infos = list(
                sorted(self._nusc_infos, key=lambda e: e["timestamp"]))
            self._nusc_infos = self._nusc_infos[::5]


@register_dataset
class NuScenesDatasetD4(NuScenesDataset):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if len(self._nusc_infos) > 28000:
            self._nusc_infos = list(
                sorted(self._nusc_infos, key=lambda e: e["timestamp"]))
            self._nusc_infos = self._nusc_infos[::4]


@register_dataset
class NuScenesDatasetD3(NuScenesDataset):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if len(self._nusc_infos) > 28000:
            self._nusc_infos = list(
                sorted(self._nusc_infos, key=lambda e: e["timestamp"]))
            self._nusc_infos = self._nusc_infos[::3]


@register_dataset
class NuScenesDatasetD2(NuScenesDataset):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if len(self._nusc_infos) > 28000:
            self._nusc_infos = list(
                sorted(self._nusc_infos, key=lambda e: e["timestamp"]))
            self._nusc_infos = self._nusc_infos[::2]


@register_dataset
class NuScenesDatasetD2Velo(NuScenesDatasetD2):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self._with_velocity = True


def _second_det_to_nusc_box(detection):
    from second.data.nuscenes.utils.data_classes import Box
    import pyquaternion
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()
    box3d[:, 6] = -box3d[:, 6] - np.pi / 2
    box_list = []
    for i in range(box3d.shape[0]):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box3d[i, 6])
        velocity = (np.nan, np.nan, np.nan)
        if box3d.shape[1] == 9:
            velocity = (*box3d[i, 7:9], 0.0)
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = Box(
            box3d[i, :3],
            box3d[i, 3:6],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def _lidar_nusc_box_to_global(info, boxes, classes, eval_version="cvpr_2019"):
    import pyquaternion
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        from second.data.nuscenes.eval.detection.config import eval_detection_configs
        # filter det in ego.
        cls_range_map = eval_detection_configs[eval_version]["class_range"]
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list


def _get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])

        # radar_f_sd_rec = nusc.get('sample_data', sample_rec['data']["RADAR_FRONT"])
        # radar_fl_sd_rec = nusc.get('sample_data', sample_rec['data']["RADAR_FRONT_LEFT"])
        # radar_fr_sd_rec = nusc.get('sample_data', sample_rec['data']["RADAR_FRONT_RIGHT"])
        # radar_bl_sd_rec = nusc.get('sample_data', sample_rec['data']["RADAR_BACK_LEFT"])
        # radar_br_sd_rec = nusc.get('sample_data', sample_rec['data']["RADAR_BACK_RIGHT"])


        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            if not sd_rec['next'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes


def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         radar_max_sweeps=5):
    train_nusc_infos = []
    val_nusc_infos = []
    from pyquaternion import Quaternion
    for sample in prog_bar(nusc.sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        radar_front_token = sample["data"]["RADAR_FRONT"]                     #++++++
        radar_front_right_token = sample["data"]["RADAR_FRONT_RIGHT"]         #++++++
        radar_front_left_token = sample["data"]["RADAR_FRONT_LEFT"]           #++++++
        radar_back_right_token = sample["data"]["RADAR_BACK_RIGHT"]           #++++++
        radar_back_left_token = sample["data"]["RADAR_BACK_LEFT"]             #++++++

        sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])

        sd_rec_rad_f = nusc.get('sample_data',sample['data']["RADAR_FRONT"])     #++++++
        sd_rec_rad_fr = nusc.get('sample_data', sample['data']["RADAR_FRONT_RIGHT"])    #++++++
        sd_rec_rad_fl = nusc.get('sample_data', sample['data']["RADAR_FRONT_LEFT"]) #++++++
        sd_rec_rad_br = nusc.get('sample_data', sample['data']["RADAR_BACK_RIGHT"]) #++++++
        sd_rec_rad_bl = nusc.get('sample_data', sample['data']["RADAR_BACK_LEFT"]) #++++++
        cs_record = nusc.get('calibrated_sensor',sd_rec['calibrated_sensor_token'])
        cs_record_rad_f = nusc.get('calibrated_sensor', sd_rec_rad_f['calibrated_sensor_token'])  #+++++
        cs_record_rad_fr = nusc.get('calibrated_sensor', sd_rec_rad_fr['calibrated_sensor_token']) #+++++
        cs_record_rad_fl = nusc.get('calibrated_sensor', sd_rec_rad_fl['calibrated_sensor_token']) #+++++
        cs_record_rad_bl = nusc.get('calibrated_sensor', sd_rec_rad_bl['calibrated_sensor_token'])  #+++++
        cs_record_rad_br = nusc.get('calibrated_sensor', sd_rec_rad_br['calibrated_sensor_token'])   #+++++

        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])

        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_front_token)

        radar_front_path,_,_ = nusc.get_sample_data(radar_front_token)                   #++
        radar_front_right_path,_,_ = nusc.get_sample_data(radar_front_right_token)       #+++
        radar_front_left_path,_,_ = nusc.get_sample_data(radar_front_left_token)         #+++
        radar_back_left_path,_,_ = nusc.get_sample_data(radar_back_left_token)           #+++
        radar_back_right_path,_,_= nusc.get_sample_data(radar_back_right_token)         #+++

        ego_to_lidar = transform_matrix(cs_record['translation'],Quaternion(cs_record['rotation']))   #+++++
        rad_front_to_ego = transform_matrix(cs_record_rad_f['translation'],                           #++++++
                                            Quaternion(cs_record_rad_f['rotation']))                  #++++++
        radar_front_right_to_ego = transform_matrix(cs_record_rad_fr['translation'],                  #+++++++
                                                    Quaternion(cs_record_rad_fr['rotation']))
        radar_front_left_to_ego = transform_matrix(cs_record_rad_fl['translation'],                   #+++++++
                                                    Quaternion(cs_record_rad_fl['rotation']))
        radar_back_right_to_ego = transform_matrix(cs_record_rad_br['translation'],                   #+++++++
                                                    Quaternion(cs_record_rad_br['rotation']))
        radar_back_left_to_ego = transform_matrix(cs_record_rad_bl['translation'],                    #+++++++
                                                    Quaternion(cs_record_rad_bl['rotation']))

        rad_front_to_lidar = np.dot(ego_to_lidar,rad_front_to_ego)                                    #+++++++
        rad_front_right_to_lidar = np.dot(ego_to_lidar, radar_front_right_to_ego)                     #+++++++
        rad_front_left_to_lidar = np.dot(ego_to_lidar,radar_front_left_to_ego)                       #+++++++
        rad_back_left_to_lidar = np.dot(ego_to_lidar, radar_back_left_to_ego)                         #+++++++
        rad_back_right_to_lidar = np.dot(ego_to_lidar, radar_back_right_to_ego)                       #+++++++

        rad_front_to_lidar_rot = rad_front_to_lidar[:3, :3]
        rad_front_right_to_lidar_rot = rad_front_right_to_lidar[:3, :3]
        rad_front_left_to_lidar_rot = rad_front_left_to_lidar[:3, :3]
        rad_back_right_to_lidar_rot = rad_back_right_to_lidar[:3, :3]
        rad_back_left_to_lidar_rot = rad_back_left_to_lidar[:3, :3]

        rad_front_to_lidar_trans = rad_front_to_lidar[:3, 3]
        rad_front_right_to_lidar_trans = rad_front_right_to_lidar[:3, 3]
        rad_front_left_to_lidar_trans = rad_front_left_to_lidar[:3, 3]
        rad_back_right_to_lidar_trans = rad_back_right_to_lidar[:3, 3]
        rad_back_left_to_lidar_trans = rad_back_left_to_lidar[:3, 3]

        assert Path(lidar_path).exists(), (
            "you must download all trainval data, key-frame only dataset performs far worse than sweeps."
        )
        info = {
            "lidar_path": lidar_path,
            "cam_front_path": cam_path,

            "radar_front_path":radar_front_path,                   #+++++
            "radar_front_right_path":radar_front_right_path,       #+++++
            "radar_front_left_path":radar_front_left_path,         #++++++
            "radar_back_right_path":radar_back_right_path,         #+++++
            "radar_back_left_path":radar_back_left_path,           #+++++

            "token": sample["token"],
            "sweeps": [],
            "lidar2ego_translation": cs_record['translation'],
            "lidar2ego_rotation": cs_record['rotation'],
            "ego2global_translation": pose_record['translation'],
            "ego2global_rotation": pose_record['rotation'],

            "radar_sweeps": [],
            "radar_front2lidar_rotation": rad_front_to_lidar_rot,
            "radar_front_right2lidar_rotation": rad_front_right_to_lidar_rot,
            "radar_front_left2lidar_rotation": rad_front_left_to_lidar_rot,
            "radar_back_right2lidar_rotation": rad_back_right_to_lidar_rot,
            "radar_back_left2lidar_rotation": rad_back_left_to_lidar_rot,

            "radar_front2lidar_translation": rad_front_to_lidar_trans,
            "radar_front_right2lidar_translation": rad_front_right_to_lidar_trans,
            "radar_front_left2lidar_translation": rad_front_left_to_lidar_trans,
            "radar_back_right2lidar_translation": rad_back_right_to_lidar_trans,
            "radar_back_left2lidar_translation": rad_back_left_to_lidar_trans,

            "timestamp": sample["timestamp"],
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
                cs_record = nusc.get('calibrated_sensor',
                                     sd_rec['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                lidar_path = nusc.get_sample_data_path(sd_rec['token'])
                sweep = {
                    "lidar_path": lidar_path,
                    "sample_data_token": sd_rec['token'],
                    "lidar2ego_translation": cs_record['translation'],
                    "lidar2ego_rotation": cs_record['rotation'],
                    "ego2global_translation": pose_record['translation'],
                    "ego2global_rotation": pose_record['rotation'],
                    "timestamp": sd_rec["timestamp"]
                }
                l2e_r_s = sweep["lidar2ego_rotation"]
                l2e_t_s = sweep["lidar2ego_translation"]
                e2g_r_s = sweep["ego2global_rotation"]
                e2g_t_s = sweep["ego2global_translation"]
                # sweep->ego->global->ego'->lidar
                l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

                R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
                sweep["sweep2lidar_rotation"] = R.T  # points @ R.T + T
                sweep["sweep2lidar_translation"] = T
                sweeps.append(sweep)
            else:
                break
        info["sweeps"] = sweeps

        radar_sweeps=[]
        sd_rec_rad_f = nusc.get('sample_data', sample['data']["RADAR_FRONT"])  # ++++++
        sd_rec_rad_fr = nusc.get('sample_data', sample['data']["RADAR_FRONT_RIGHT"])  # ++++++
        sd_rec_rad_fl = nusc.get('sample_data', sample['data']["RADAR_FRONT_LEFT"])  # ++++++
        sd_rec_rad_br = nusc.get('sample_data', sample['data']["RADAR_BACK_RIGHT"])  # ++++++
        sd_rec_rad_bl = nusc.get('sample_data', sample['data']["RADAR_BACK_LEFT"])  # ++++++
        while len(radar_sweeps) < radar_max_sweeps:
            if not sd_rec_rad_f['prev'] == "":
                sd_rec_rad_f = nusc.get('sample_data', sd_rec_rad_f['prev'])

                radar_front_path = nusc.get_sample_data_path(sd_rec_rad_f['token'])  #+++++

                cs_record_rad_f = nusc.get('calibrated_sensor', sd_rec_rad_f['calibrated_sensor_token'])
                ego_to_lidar = transform_matrix(cs_record['translation'], Quaternion(cs_record['rotation']))  # +++++
                rad_front_to_ego = transform_matrix(cs_record_rad_f['translation'],  # ++++++
                                                    Quaternion(cs_record_rad_f['rotation']))
                rad_front_to_lidar = np.dot(ego_to_lidar, rad_front_to_ego)
                rad_front_to_lidar_rot = rad_front_to_lidar[:3, :3]
                rad_front_to_lidar_trans = rad_front_to_lidar[:3, 3]

            if not sd_rec_rad_fr['prev'] == "":
                sd_rec_rad_fr = nusc.get('sample_data', sd_rec_rad_fr['prev'])
                radar_front_right_path = nusc.get_sample_data_path(sd_rec_rad_fr['token'])  # +++
                radar_front_right_to_ego = transform_matrix(cs_record_rad_fr['translation'],  # +++++++
                                                            Quaternion(cs_record_rad_fr['rotation']))
                rad_front_right_to_lidar = np.dot(ego_to_lidar, radar_front_right_to_ego)  # +++++++
                rad_front_right_to_lidar_rot = rad_front_right_to_lidar[:3, :3]
                rad_front_right_to_lidar_trans = rad_front_right_to_lidar[:3, 3]

            if not sd_rec_rad_fl['prev'] == "":
                sd_rec_rad_fl = nusc.get('sample_data', sd_rec_rad_fl['prev'])
                radar_front_left_path = nusc.get_sample_data_path(sd_rec_rad_fl['token'])  # +++
                radar_front_left_to_ego = transform_matrix(cs_record_rad_fl['translation'],  # +++++++
                                                           Quaternion(cs_record_rad_fl['rotation']))
                rad_front_left_to_lidar = np.dot(ego_to_lidar, radar_front_left_to_ego)  # +++++++
                rad_front_left_to_lidar_rot = rad_front_left_to_lidar[:3, :3]
                rad_front_left_to_lidar_trans = rad_front_left_to_lidar[:3, 3]

            if not sd_rec_rad_br['prev'] == "":
                sd_rec_rad_br = nusc.get('sample_data', sd_rec_rad_br['prev'])
                radar_back_right_path = nusc.get_sample_data_path(sd_rec_rad_br['token'])  # +++
                radar_back_right_to_ego = transform_matrix(cs_record_rad_br['translation'],  # +++++++
                                                           Quaternion(cs_record_rad_br['rotation']))
                rad_back_right_to_lidar = np.dot(ego_to_lidar, radar_back_right_to_ego)
                rad_back_right_to_lidar_trans = rad_back_right_to_lidar[:3, 3]

            if not sd_rec_rad_bl['prev'] == "":
                sd_rec_rad_bl = nusc.get('sample_data', sd_rec_rad_bl['prev'])
                radar_back_left_path = nusc.get_sample_data_path(sd_rec_rad_bl['token'])  # +++
                radar_back_left_to_ego = transform_matrix(cs_record_rad_bl['translation'],  # +++++++
                                                          Quaternion(cs_record_rad_bl['rotation']))
                rad_back_left_to_lidar = np.dot(ego_to_lidar, radar_back_left_to_ego)
                rad_back_left_to_lidar_rot = rad_back_left_to_lidar[:3, :3]
                rad_back_left_to_lidar_trans = rad_back_left_to_lidar[:3, 3]



                radar_sweep = {
                    "radar_front_path":radar_front_path,
                    "radar_front_right_path": radar_front_right_path,
                    "radar_front_left_path": radar_front_left_path,
                    "radar_back_left_path":radar_back_left_path,
                    "radar_back_right_path":radar_back_right_path,

                    "radar_f_sample_data_token": sd_rec_rad_f['token'],
                    "radar_fr_sample_data_token": sd_rec_rad_fr['token'],
                    "radar_fl_sample_data_token": sd_rec_rad_fl['token'],
                    "radar_bl_sample_data_token": sd_rec_rad_bl['token'],
                    "radar_br_sample_data_token": sd_rec_rad_br['token'],

                    "radar_front2lidar_rotation": rad_front_to_lidar_rot,
                    "radar_front_right2lidar_rotation": rad_front_right_to_lidar_rot,
                    "radar_front_left2lidar_rotation": rad_front_left_to_lidar_rot,
                    "radar_back_right2lidar_rotation": rad_back_right_to_lidar_rot,
                    "radar_back_left2lidar_rotation": rad_back_left_to_lidar_rot,

                    "radar_front2lidar_translation": rad_front_to_lidar_trans,
                    "radar_front_right2lidar_translation": rad_front_right_to_lidar_trans,
                    "radar_front_left2lidar_translation": rad_front_left_to_lidar_trans,
                    "radar_back_right2lidar_translation": rad_back_right_to_lidar_trans,
                    "radar_back_left2lidar_translation": rad_back_left_to_lidar_trans,
                    "timestamp": sd_rec_rad_f["timestamp"]
                }
                radar_sweeps.append(radar_sweep)
            else:
                break
        info["radar_sweeps"] = radar_sweeps
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NuScenesDataset.NameMapping:
                    names[i] = NuScenesDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            # change the rot format will break all checkpoint, so...
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array(
                [a["num_radar_pts"] for a in annotations])
        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)
    return train_nusc_infos, val_nusc_infos


def create_nuscenes_infos(root_path, version="v1.0-trainval", max_sweeps=10, radar_max_sweeps=5):
    from nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from second.data.nuscenes.utils import splits
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train #根据场景号进行索引
        val_scenes = splits.mini_val
    else:
        raise ValueError("unknown")
    test = "test" in version
    root_path = Path(root_path)
    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)        #可用的情景
    available_scene_names = [s["name"] for s in available_scenes]   # <class 'list'>: ['scene-0061', 'scene-0103', 'scene-0553', 'scene-0655', 'scene-0757', 'scene-0796', 'scene-0916', 'scene-1077', 'scene-1094', 'scene-1100']
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]["token"]
        for s in val_scenes
    ])
    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(
            f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps, radar_max_sweeps=radar_max_sweeps)   #将info信息加载进去
    metadata = {
        "version": version,
    }
    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        data = {
            "infos": train_nusc_infos,
            "metadata": metadata,
        }
        with open(root_path / "infos_test.pkl", 'wb') as f:
            pickle.dump(data, f)
    else:
        print(
            f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
        )
        data = {
            "infos": train_nusc_infos,
            "metadata": metadata,
        }
        with open(root_path / "infos_trainD1.pkl", 'wb') as f:
            pickle.dump(data, f)
        data["infos"] = val_nusc_infos
        with open(root_path / "infos_valD1.pkl", 'wb') as f:
            pickle.dump(data, f)


def get_box_mean(info_path, class_name="vehicle.car",
                 eval_version="cvpr_2019"):
    with open(info_path, 'rb') as f:
        nusc_infos = pickle.load(f)["infos"]
    from second.data.nuscenes.eval.detection.config import eval_detection_configs
    cls_range_map = eval_detection_configs[eval_version]["class_range"]

    gt_boxes_list = []
    gt_vels_list = []
    for info in nusc_infos:
        gt_boxes = info["gt_boxes"]
        gt_vels = info["gt_velocity"]
        gt_names = info["gt_names"]
        mask = np.array([s == class_name for s in info["gt_names"]],
                        dtype=np.bool_)
        gt_names = gt_names[mask]
        gt_boxes = gt_boxes[mask]
        gt_vels = gt_vels[mask]
        det_range = np.array([cls_range_map[n] for n in gt_names])
        det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
        mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
        mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)

        gt_boxes_list.append(gt_boxes[mask].reshape(-1, 7))
        gt_vels_list.append(gt_vels[mask].reshape(-1, 2))
    gt_boxes_list = np.concatenate(gt_boxes_list, axis=0)
    gt_vels_list = np.concatenate(gt_vels_list, axis=0)
    nan_mask = np.isnan(gt_vels_list[:, 0])
    gt_vels_list = gt_vels_list[~nan_mask]

    # return gt_vels_list.mean(0).tolist()
    return {
        "box3d": gt_boxes_list.mean(0).tolist(),
        "detail": gt_boxes_list
        # "velocity": gt_vels_list.mean(0).tolist(),
    }


def get_all_box_mean(info_path):
    det_names = set()
    for k, v in NuScenesDataset.NameMapping.items():
        if v not in det_names:
            det_names.add(v)
    det_names = sorted(list(det_names))
    res = {}
    details = {}
    for k in det_names:
        result = get_box_mean(info_path, k)
        details[k] = result["detail"]
        res[k] = result["box3d"]
    print(json.dumps(res, indent=2))
    return details


def render_nusc_result(nusc, results, sample_token):
    from second.data.nuscenes.utils.data_classes import Box
    from pyquaternion import Quaternion
    annos = results[sample_token]
    sample = nusc.get("sample", sample_token)
    sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    boxes = []
    for anno in annos:
        rot = Quaternion(anno["rotation"])
        box = Box(
            anno["translation"],
            anno["size"],
            rot,
            name=anno["detection_name"])
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        #  Move box to sensor coord system
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes.append(box)
    nusc.explorer.render_sample_data(
        sample["data"]["LIDAR_TOP"], extern_boxes=boxes, nsweeps=10)
    nusc.explorer.render_sample_data(sample["data"]["LIDAR_TOP"], nsweeps=10)


def cluster_trailer_box(info_path, class_name="bus"):
    with open(info_path, 'rb') as f:
        nusc_infos = pickle.load(f)["infos"]
    from second.data.nuscenes.eval.detection.config import eval_detection_configs
    cls_range_map = eval_detection_configs["cvpr_2019"]["class_range"]
    gt_boxes_list = []
    for info in nusc_infos:
        gt_boxes = info["gt_boxes"]
        gt_names = info["gt_names"]
        mask = np.array([s == class_name for s in info["gt_names"]],
                        dtype=np.bool_)
        gt_names = gt_names[mask]
        gt_boxes = gt_boxes[mask]
        det_range = np.array([cls_range_map[n] for n in gt_names])
        det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
        mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
        mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)

        gt_boxes_list.append(gt_boxes[mask].reshape(-1, 7))
    gt_boxes_list = np.concatenate(gt_boxes_list, axis=0)
    trailer_dims = gt_boxes_list[:, 3:6]
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(trailer_dims)
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print(n_clusters_, n_noise_)
    print(trailer_dims)

    import matplotlib.pyplot as plt
    unique_labels = set(labels)
    colors = [
        plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
    ]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]
        class_member_mask = (labels == k)
        xy = trailer_dims[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            'o',
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=14)
        xy = trailer_dims[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            'o',
            markerfacecolor=tuple(col),
            markeredgecolor='k',
            markersize=6)
    plt.show()

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm

def from_file(
          file_name: str,
          invalid_states: List[int] = None,
          dynprop_states: List[int] = None,
          ambig_states: List[int] = None):
    """
    Loads RADAR data from a Point Cloud Data file. See details below.
    :param file_name: The path of the pointcloud file.
    :param invalid_states: Radar states to be kept. See details below.
    :param dynprop_states: Radar states to be kept. Use [0, 2, 6] for moving objects only. See details below.
    :param ambig_states: Radar states to be kept. See details below.
    To keep all radar returns, set each state filter to range(18).
    :return: <np.float: d, n>. Point cloud matrix with d dimensions and n points.

    Example of the header fields:
    # .PCD v0.7 - Point Cloud Data file format
    VERSION 0.7
    FIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms
    SIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1
    TYPE F F F I I F F F F F I I I I I I I I
    COUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    WIDTH 125
    HEIGHT 1
    VIEWPOINT 0 0 0 1 0 0 0
    POINTS 125
    DATA binary

    Below some of the fields are explained in more detail:

    x is front, y is left

    vx, vy are the velocities in m/s.
    vx_comp, vy_comp are the velocities in m/s compensated by the ego motion.
    We recommend using the compensated velocities.

    invalid_state: state of Cluster validity state.
    (Invalid states)
    0x01	invalid due to low RCS
    0x02	invalid due to near-field artefact
    0x03	invalid far range cluster because not confirmed in near range
    0x05	reserved
    0x06	invalid cluster due to high mirror probability
    0x07	Invalid cluster because outside sensor field of view
    0x0d	reserved
    0x0e	invalid cluster because it is a harmonics
    (Valid states)
    0x00	valid
    0x04	valid cluster with low RCS
    0x08	valid cluster with azimuth correction due to elevation
    0x09	valid cluster with high child probability
    0x0a	valid cluster with high probability of being a 50 deg artefact
    0x0b	valid cluster but no local maximum
    0x0c	valid cluster with high artefact probability
    0x0f	valid cluster with above 95m in near range
    0x10	valid cluster with high multi-target probability
    0x11	valid cluster with suspicious angle

    dynProp: Dynamic property of cluster to indicate if is moving or not.
    0: moving
    1: stationary
    2: oncoming
    3: stationary candidate
    4: unknown
    5: crossing stationary
    6: crossing moving
    7: stopped

    ambig_state: State of Doppler (radial velocity) ambiguity solution.
    0: invalid
    1: ambiguous
    2: staggered ramp
    3: unambiguous
    4: stationary candidates

    pdh0: False alarm probability of cluster (i.e. probability of being an artefact caused by multipath or similar).
    0: invalid
    1: <25%
    2: 50%
    3: 75%
    4: 90%
    5: 99%
    6: 99.9%
    7: <=100%
    """

    assert file_name.endswith('.pcd'), 'Unsupported filetype {}'.format(file_name)

    meta = []
    with open(file_name, 'rb') as f:
        for line in f:
            line = line.strip().decode('utf-8')
            meta.append(line)
            if line.startswith('DATA'):
                break

        data_binary = f.read()

    # Get the header rows and check if they appear as expected.
    assert meta[0].startswith('#'), 'First line must be comment'
    assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
    sizes = meta[3].split(' ')[1:]
    types = meta[4].split(' ')[1:]
    counts = meta[5].split(' ')[1:]
    width = int(meta[6].split(' ')[1])
    height = int(meta[7].split(' ')[1])
    data = meta[10].split(' ')[1]
    feature_count = len(types)
    assert width > 0
    assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
    assert height == 1, 'Error: height != 0 not supported!'
    assert data == 'binary'

    # Lookup table for how to decode the binaries.
    unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                     'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                     'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
    types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

    # Decode each point.
    offset = 0
    # if len(data_binary)==5375:
    #     point_count = 125
    # if len(data_binary) == 10750:
    #     point_count =250
    # if len(data_binary)==16125:
    #     point_count = 375
    # if len(data_binary) == 21500:
    #     point_count =500
    point_count= width
    points = []
    # print(len(data_binary))
    for i in range(point_count):
        point = []
        for p in range(feature_count):
            start_p = offset
            end_p = start_p + int(sizes[p])

            assert end_p <= len(data_binary)
            point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
            point.append(point_p)
            offset = end_p
            # print(offset)
        points.append(point)

    # A NaN in the first point indicates an empty pointcloud.
    point = np.array(points[0])
    if np.any(np.isnan(point)):
        return (np.zeros((feature_count, 0)))

    # Convert to numpy matrix.
    points = np.array(points).transpose()

    # If no parameters are provided, use default settings.
    invalid_states = invalid_states if invalid_states is None else invalid_states
    dynprop_states = dynprop_states if dynprop_states is None else dynprop_states
    ambig_states = ambig_states if ambig_states is None else ambig_states

    # Filter points with an invalid state.
    valid = [p in invalid_states for p in points[-4, :]]
    points = points[:, valid]

    # Filter by dynProp.
    valid = [p in dynprop_states for p in points[3, :]]
    points = points[:, valid]

    # Filter by ambig_state.
    valid = [p in ambig_states for p in points[11, :]]
    points = points[:, valid]

    return points.T

if __name__ == "__main__":
    fire.Fire()
