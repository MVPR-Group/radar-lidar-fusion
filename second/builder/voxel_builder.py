import numpy as np

from spconv.utils import VoxelGeneratorV2
from second.protos import voxel_generator_pb2


def build(voxel_config):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    if not isinstance(voxel_config, (voxel_generator_pb2.VoxelGenerator)):
        raise ValueError('input_reader_config not of type '
                         'input_reader_pb2.InputReader.')
    voxel_generator = VoxelGeneratorV2(                                #VoxelGeneratorV2系统函数
        voxel_size=list(voxel_config.voxel_size),         #[0.25,0.25,20]
        point_cloud_range=list(voxel_config.point_cloud_range),    #[-50.0, -50.0, -10.0, 50.0, 50.0, 10.0]
        max_num_points=voxel_config.max_number_of_points_per_voxel,    #60
        max_voxels=20000,
        full_mean=voxel_config.full_empty_part_with_mean,    #false
        block_filtering=voxel_config.block_filtering,        #false
        block_factor=voxel_config.block_factor,              #0
        block_size=voxel_config.block_size,                  #0
        height_threshold=voxel_config.height_threshold)      #0.0
    return voxel_generator
