import numpy as np
import matplotlib.pyplot as plt
import pickle
import struct
from typing import Tuple, List, Dict
from pathlib import Path
import fire
import torch
from google.protobuf import text_format
from second.utils import simplevis
from second.pytorch.train import build_network
from second.protos import pipeline_pb2
from second.utils import config_tool


def from_file(
          file_name: str,
          invalid_states: List[int] = None,
          dynprop_states: List[int] = None,
          ambig_states: List[int] = None):


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

if __name__ == '__main__':
    for num in range(60,80):
        config_path = "/home/lichao/second-master_D8r/second/configs/nuscenes/all.pp.lowa.config"

        config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, config)
        input_cfg = config.eval_input_reader
        model_cfg = config.model.second
        # config_tool.change_detection_range(model_cfg, [-50, -50, 50, 50])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        ckpt_path = "/home/lichao/model_lowa_add/voxelnet-140000.tckpt"
        net = build_network(model_cfg).to(device).eval()
        net.load_state_dict(torch.load(ckpt_path))
        target_assigner = net.target_assigner
        voxel_generator = net.voxel_generator


        grid_size = voxel_generator.grid_size
        feature_map_size = grid_size[:2] // config_tool.get_downsample_factor(model_cfg)
        feature_map_size = [*feature_map_size, 1][::-1]


        #Here is the thing I changed
        # v_path = '/home/lichao/v1.0-mini/samples/LIDAR_TOP/n015-2018-10-08-15-36-50+0800__LIDAR_TOP__1538984251947010.pcd.bin'
        anchors = target_assigner.generate_anchors(feature_map_size)["anchors"]
        anchors = torch.tensor(anchors, dtype=torch.float32, device=device)
        anchors = anchors.view(1, -1, 7)

        # points = np.fromfile(
        #     v_path, dtype=np.float32, count=-1).reshape([-1, 5])

        info_path = input_cfg.dataset.kitti_info_path
        with open(info_path, 'rb') as f:
            info = pickle.load(f)

        info = info['infos']
        # info = info[3066]
        info = info[num]
        lidar_path = Path(info['lidar_path'])
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
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                "sweep2lidar_rotation"].T
            points_sweep[:, :3] += sweep["sweep2lidar_translation"]
            points_sweep[:, 4] = ts - sweep_ts
            sweep_points_list.append(points_sweep)

        points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]
        all_radar_points = []
        radar_name = ['radar_front_path', 'radar_front_right_path', 'radar_front_left_path', 'radar_back_left_path',
                      'radar_back_right_path']
        radar_rotation_name = ['radar_front2lidar_rotation', 'radar_front_right2lidar_rotation',
                               'radar_front_left2lidar_rotation', 'radar_back_left2lidar_rotation',
                               'radar_back_left2lidar_rotation']
        radar_translation_name = ['radar_front2lidar_translation', 'radar_front_right2lidar_translation',
                                  'radar_front_left2lidar_translation', 'radar_back_left2lidar_translation',
                                  'radar_back_right2lidar_translation']
        for i in range(5):
            radar_path = Path(info[str(radar_name[i])])
            radar_points = from_file(str(radar_path), invalid_states=[0],
                                     dynprop_states=list(range(7)),
                                     ambig_states=[3])
            if radar_points.size != 0:
                radar_points[:, :3] = radar_points[:, :3] @ info[str(radar_rotation_name[i])]
                radar_points[:, :3] -= info[str(radar_translation_name[i])]
            else:
                radar_points = np.zeros([1, 18])
            radar_points = radar_points.reshape([-1, 18])
            radar_sweeps_points_list = [radar_points]
            for radar_sweeps in info["radar_sweeps"]:
                radar_points_sweeps = from_file(str(radar_sweeps[str(radar_name[i])]), invalid_states=[0],
                                                dynprop_states=list(range(7)),
                                                ambig_states=[3])
                if radar_points_sweeps.size == 0:
                    radar_points_sweeps = np.zeros([1, 18])
                radar_points_sweeps = radar_points_sweeps.reshape([-1, 18])
                radar_points_sweeps[:, :3] = radar_points_sweeps[:, :3] @ radar_sweeps[
                    str(radar_rotation_name[i])]
                radar_points_sweeps[:, :3] -= radar_sweeps[str(radar_translation_name[i])]
                radar_sweeps_points_list.append(radar_points_sweeps)
            try:
                radar_points = np.concatenate(radar_sweeps_points_list, axis=0)[:,
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
            except:
                print(radar_sweeps_points_list.shape)
            # radar_points.append(radar_points)     #3150个点还多了？
            all_radar_points.append(radar_points)
        all_radar_points = np.concatenate(all_radar_points, axis=0)[:, [0, 1, 2, 5, 9, 10]]

        points = points[:, :4]
        points[:, 3] = points[:, 3] / 255.0
        dic = voxel_generator.generate(points)
        dic_radar = voxel_generator.generate(all_radar_points)
        voxels,coords, num_points = dic['voxels'], dic['coordinates'], dic['num_points_per_voxel']
        radar_voxels,radar_coords,radar_num_points=dic_radar['voxels'],dic_radar['coordinates'],dic_radar['num_points_per_voxel']
        print(voxels.shape)
        # add batch idx to coords
        coords = np.pad(coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        radar_coords = np.pad(radar_coords, ((0, 0), (1, 0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=device)
        radar_voxels = torch.tensor(radar_voxels, dtype=torch.float32, device=device)
        coords = torch.tensor(coords, dtype=torch.int32, device=device)
        radar_coords = torch.tensor(radar_coords, dtype=torch.int32, device=device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=device)
        radar_num_points = torch.tensor(radar_num_points, dtype=torch.int32, device=device)


        example = {
            "anchors": anchors,
            "voxels": voxels,
            "radar_voxels":radar_voxels,
            "num_points": num_points,
            "radar_num_points":radar_num_points,
            "coordinates": coords,
            "radar_coordinates":radar_coords
        }
        pred = net(example)[0]

        # print(pred)

        boxes_lidar = pred["box3d_lidar"].detach().cpu().numpy()
        box3d = pred["box3d_lidar"].detach().cpu().numpy()
        # box3d = info['gt_boxes']
        scores = pred["scores"].detach().cpu().numpy()
        labels = pred["label_preds"].detach().cpu().numpy()
        idx = np.where(scores >0.3)[0]
        # print(idx)
        # filter low score ones
        # print(boxes_lidar)
        box3d = box3d[idx, :]
        # print(labels)
        # label is one-dim
        labels = np.take(labels, idx)
        # print(labels)
        scores = np.take(scores, idx)
        # print(scores)
        vis_voxel_size = [0.1, 0.1, 0.1]
        vis_point_range = [-50, -50, -3, 50, 50, 1]
        # bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
        bev_map = simplevis.point_to_vis_bev(points, vis_voxel_size, vis_point_range)
        bev_map = simplevis.draw_box_in_bev(bev_map, vis_point_range, box3d, [0, 255, 0], 2)

        plt.imshow(bev_map)
        # plt.savefig("/home/lichao/image_rain_label2/%d.png"%(num))
        # plt.savefig("4.3.png")
        plt.savefig("/home/lichao/lidar_image_0.3/%d.png"%(num))


