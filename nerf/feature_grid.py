""" manually written sort-of-low-level implementation for voxel-based 3D volumetric representations """
from typing import Tuple, NamedTuple, Optional, Callable, Dict, Any, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import grid_sample, interpolate

import numpy as np

# keys used in saved_configuration
THRE3D_REPR = "thre3d_repr"
RENDER_PROCEDURE = "render_procedure"
RENDER_CONFIG = "render_config"
RENDER_CONFIG_TYPE = "render_config_type"
STATE_DICT = "state_dict"
CONFIG_DICT = "config_dict"

# specific to voxel_grids:
u_FEATURES = "_features"


def adjust_dynamic_range(
    data: Union[np.array, Tensor],
    drange_in: Tuple[float, float],
    drange_out: Tuple[float, float],
    slack: bool = False,
) -> np.array:
    """
    converts the data from the range `drange_in` into `drange_out`
    Args:
        data: input data array
        drange_in: data range [total_min_val, total_max_val]
        drange_out: output data range [min_val, max_val]
        slack: whether to cut some slack in range adjustment :D
    Returns: range_adjusted_data
    """
    if drange_in != drange_out:
        if slack:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0])
            )
            bias = np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale
            data = data * scale + bias
        else:
            old_min, old_max = np.float32(drange_in[0]), np.float32(drange_in[1])
            new_min, new_max = np.float32(drange_out[0]), np.float32(drange_out[1])
            data = (
                (data - old_min) / (old_max - old_min) * (new_max - new_min)
            ) + new_min
            data = data.clip(drange_out[0], drange_out[1])
    return data


class VoxelSize(NamedTuple):
    """lengths of a single voxel's edges in the x, y and z dimensions
    allows for the possibility of anisotropic voxels"""

    x_size: float = 1.0
    y_size: float = 1.0
    z_size: float = 1.0


class VoxelGridLocation(NamedTuple):
    """indicates where the Voxel-Grid is located in World Coordinate System
    i.e. indicates where the centre of the grid is located in the World
    The Grid is always assumed to be axis aligned"""

    x_coord: float = 0.0
    y_coord: float = 0.0
    z_coord: float = 0.0


class AxisAlignedBoundingBox(NamedTuple):
    """defines an axis-aligned voxel grid's spatial extent"""

    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]


class VoxelGrid(Module):
    def __init__(
        self,
        # grid values:
        features: Tensor,
        # grid coordinate-space properties:
        voxel_size: VoxelSize,
        grid_location: Optional[VoxelGridLocation] = VoxelGridLocation(),
        # feature activations:
        feature_preactivation: Callable[[Tensor], Tensor] = torch.nn.Identity(),
        feature_postactivation: Callable[[Tensor], Tensor] = torch.nn.Identity(),
        # radiance function / transfer function:
        radiance_transfer_function: Callable[[Tensor, Tensor], Tensor] = None,
        tunable: bool = False,
    ):
        """
        Defines a Voxel-Grid denoting a 3D-volume. To obtain features of a particular point inside
        the volume, we obtain continuous features by doing trilinear interpolation.
        Args:
            features: Tensor of shape [W x D x H x F] corresponds to the features on the grid-vertices
            voxel_size: Size of each voxel. (could be different in different axis (x, y, z))
            grid_location: Location of the center of the grid
            feature_preactivation: the activation to be applied to the features before interpolating.
            feature_postactivation: the activation to be applied to the features after interpolating.
            radiance_transfer_function: the function that maps (can map)
                                        the interpolated features to RGB (radiance) values
            tunable: whether to treat the features Tensors as tunable (trainable) parameters
        """
        # as usual start with assertions about the inputs:
        assert (
            len(features.shape) == 4
        ), f"features should be of shape [W x D x H x F] as opposed to ({features.shape})"
        super().__init__()

        # initialize the state of the object
        self._features = features
        self._feature_preactivation = feature_preactivation
        self._feature_postactivation = feature_postactivation
        self._radiance_transfer_function = radiance_transfer_function
        self._grid_location = grid_location
        self._voxel_size = voxel_size
        self._tunable = tunable

        if tunable:
            self._features = torch.nn.Parameter(self._features)

        # features can be used:
        self._device = features.device

        # note the x, y and z conventions for the width (+ve right), depth (+ve inwards) and height (+ve up)
        self.width_x, self.depth_y, self.height_z = (
            self._features.shape[0],
            self._features.shape[1],
            self._features.shape[2],
        )

        # setup the bounding box planes
        self._aabb = self._setup_bounding_box_planes()

    @property
    def features(self) -> Tensor:
        return self._features

    @features.setter
    def features(self, features: Tensor) -> None:
        assert (
            features.shape == self._features.shape
        ), f"new features don't match original feature tensor's dimensions"
        if self._tunable and not isinstance(features, torch.nn.Parameter):
            self._features = torch.nn.Parameter(features)
        else:
            self._features = features

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        return self._aabb

    @property
    def grid_dims(self) -> Tuple[int, int, int]:
        return self.width_x, self.depth_y, self.height_z

    @property
    def voxel_size(self) -> VoxelSize:
        return self._voxel_size

    @voxel_size.setter
    def voxel_size(self, voxel_size: VoxelSize) -> None:
        self._voxel_size = voxel_size

    def get_save_config_dict(self) -> Dict[str, Any]:
        save_config_dict = self.get_config_dict()
        save_config_dict.update({"voxel_size": self._voxel_size})
        return save_config_dict

    def get_config_dict(self) -> Dict[str, Any]:
        return {
            "grid_location": self._grid_location,
            "feature_preactivation": self._feature_preactivation,
            "feature_postactivation": self._feature_postactivation,
            "radiance_transfer_function": self._radiance_transfer_function,
            "tunable": self._tunable,
        }

    def _setup_bounding_box_planes(self) -> AxisAlignedBoundingBox:
        # compute half grid dimensions
        half_width = (self.width_x * self._voxel_size.x_size) / 2
        half_depth = (self.depth_y * self._voxel_size.y_size) / 2
        half_height = (self.height_z * self._voxel_size.z_size) / 2

        # compute the AABB (bounding_box_planes)
        width_x_range = (
            self._grid_location.x_coord - half_width,
            self._grid_location.x_coord + half_width,
        )
        depth_y_range = (
            self._grid_location.y_coord - half_depth,
            self._grid_location.y_coord + half_depth,
        )
        height_z_range = (
            self._grid_location.z_coord - half_height,
            self._grid_location.z_coord + half_height,
        )

        # return the computed planes in the packed AABB datastructure:
        return AxisAlignedBoundingBox(
            x_range=width_x_range,
            y_range=depth_y_range,
            z_range=height_z_range,
        )

    def _normalize_points(self, points: Tensor) -> Tensor:
        normalized_points = torch.empty_like(points, device=points.device)
        for coordinate_index, coordinate_range in enumerate(self._aabb):
            normalized_points[:, coordinate_index] = adjust_dynamic_range(
                points[:, coordinate_index],
                drange_in=coordinate_range,
                drange_out=(-1.0, 1.0),
                slack=True,
            )
        return normalized_points

    def extra_repr(self) -> str:
        return (
            f"grid_dims: {(self.width_x, self.depth_y, self.height_z)}, "
            f"feature_dims: {self._features.shape[-1]}, "
            f"voxel_size: {self._voxel_size}, "
            f"grid_location: {self._grid_location}, "
            f"tunable: {self._tunable}"
        )

    def get_bounding_volume_vertices(self) -> Tensor:
        x_min, x_max = self._aabb.x_range
        y_min, y_max = self._aabb.y_range
        z_min, z_max = self._aabb.z_range
        return torch.tensor(
            [
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max],
            ],
            dtype=torch.float32,
        )

    def test_inside_volume(self, points: Tensor) -> Tensor:
        """
        tests whether the points are inside the AABB or not
        Args:
            points: Tensor of shape [N x 3 (NUM_COORD_DIMENSIONS)]
        Returns: Tensor of shape [N x 1]  (boolean)
        """
        return torch.logical_and(
            torch.logical_and(
                torch.logical_and(
                    points[..., 0:1] > self._aabb.x_range[0],
                    points[..., 0:1] < self._aabb.x_range[1],
                ),
                torch.logical_and(
                    points[..., 1:2] > self._aabb.y_range[0],
                    points[..., 1:2] < self._aabb.y_range[1],
                ),
            ),
            torch.logical_and(
                points[..., 2:] > self._aabb.z_range[0],
                points[..., 2:] < self._aabb.z_range[1],
            ),
        )

    def forward(self, points: Tensor, viewdirs: Optional[Tensor] = None) -> Tensor:
        """
        computes the features/radiance at the requested 3D points
        Args:
            points: Tensor of shape [N x 3 (NUM_COORD_DIMENSIONS)]
            viewdirs: Tensor of shape [N x 3 (NUM_COORD_DIMENSIONS)]
                      this tensor represents viewing directions in world-coordinate-system
        Returns: either Tensor of shape [N x <3 + 1> (NUM_COLOUR_CHANNELS)]
                 or of shape [N x <features + 1> (number of features)], depending upon
                 whether the `self._radiance_transfer_function` is None.
        """
        # obtain the range-normalized points for interpolation
        normalized_points = self._normalize_points(points)

        # interpolate and compute features
        preactivated_features = self._feature_preactivation(self._features)
        interpolated_features = (
            grid_sample(
                preactivated_features[None, ...].permute(0, 4, 3, 2, 1),
                normalized_points[None, None, None, ...],
                align_corners=False,
            )
            .permute(0, 2, 3, 4, 1)
            .squeeze()
        )
        interpolated_features = self._feature_postactivation(interpolated_features)

        # apply the radiance transfer function if it is not None and if view-directions are available
        if self._radiance_transfer_function is not None and viewdirs is not None:
            interpolated_features = self._radiance_transfer_function(
                interpolated_features, viewdirs
            )

        # return a unified tensor containing interpolated features
        return interpolated_features


def scale_voxel_grid_with_required_output_size(
    voxel_grid: VoxelGrid, output_size: Tuple[int, int, int], mode: str = "trilinear"
) -> VoxelGrid:

    # extract relevant information from the original input voxel_grid:
    og_unified_feature_tensor = voxel_grid.features
    og_voxel_size = voxel_grid.voxel_size

    # compute the new features using pytorch's interpolate function
    new_features = interpolate(
        og_unified_feature_tensor.permute(3, 0, 1, 2)[None, ...],
        size=output_size,
        mode=mode,
        align_corners=False,  # never use align_corners=True :D
        recompute_scale_factor=False,  # this needs to be set for some reason, I can't remember :D
    )[0]
    new_features = new_features.permute(1, 2, 3, 0)

    # a paranoid check that the interpolated features have the exact same output_size as required
    assert new_features.shape[:-1] == output_size, f"newfeature shape: {new_features.shape[:-1]} and output size: {output_size} mismatch!"

    # new voxel size is also similarly scaled
    new_voxel_size = VoxelSize(
        (og_voxel_size.x_size * voxel_grid.width_x) / output_size[0],
        (og_voxel_size.y_size * voxel_grid.depth_y) / output_size[1],
        (og_voxel_size.z_size * voxel_grid.height_z) / output_size[2],
    )

    # create a new voxel_grid by cloning the input voxel_grid and update the newly scaled properties
    new_voxel_grid = VoxelGrid(
        features=new_features,
        voxel_size=new_voxel_size,
        **voxel_grid.get_config_dict(),
    )

    # noinspection PyProtectedMember
    return new_voxel_grid


def create_voxel_grid_from_saved_info_dict(saved_info: Dict[str, Any]) -> VoxelGrid:
    features = torch.empty_like(saved_info[THRE3D_REPR][STATE_DICT][u_FEATURES])
    voxel_grid = VoxelGrid(
        features=features, **saved_info[THRE3D_REPR][CONFIG_DICT]
    )
    voxel_grid.load_state_dict(saved_info[THRE3D_REPR][STATE_DICT])
    return voxel_grid


def compute_thre3d_grid_sizes(
    final_required_resolution: Tuple[int, int, int],
    num_stages: int,
    scale_factor: float,
):
    x, y, z = final_required_resolution
    grid_sizes = [(x, y, z)]
    for _ in range(num_stages - 1):
        x = int(np.ceil((1 / scale_factor) * x))
        y = int(np.ceil((1 / scale_factor) * y))
        z = int(np.ceil((1 / scale_factor) * z))
        grid_sizes.insert(0, (x, y, z))
    return grid_sizes


class MultiResVoxelGrid(Module):
    def __init__(self, bounds, finest_grid_dims=(1024, 1024, 1), feature_dim_per_level=2, level_num=8):
        super().__init__()
        self.bounds = bounds
        self.level_num = level_num
        self.feature_dim_per_level = feature_dim_per_level
        self.level_grid_dims = [(int(finest_grid_dims[0]*(.5**i)), int(finest_grid_dims[1]*(.5**i)), 1) for i in range(level_num)]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.level_voxel_grid = nn.ModuleList()

        for i in range(level_num):
            features = torch.empty((*self.level_grid_dims[i], feature_dim_per_level), dtype=torch.float32, device=self.device)
            torch.nn.init.uniform_(features, -1., 1.)
            voxel_size = VoxelSize(*[bound / grid_dim for bound, grid_dim in zip(bounds, self.level_grid_dims[i])])
            voxel_grid = VoxelGrid(features=features, feature_postactivation=torch.nn.Tanh(), voxel_size=voxel_size, grid_location=VoxelGridLocation(0., 0., 0.), tunable=True)
            self.level_voxel_grid.append(voxel_grid)
        
        self.out_dim = feature_dim_per_level * level_num
    
    def forward(self, x):
        grid_features = []
        for i in range(self.level_num):
            grid_features.append(self.level_voxel_grid[i](x))
        grid_features = torch.cat(grid_features, dim=-1)
        return grid_features
    
    def test_inside_volume(self, x):
        inside_points_mask = self.level_voxel_grid[0].test_inside_volume(x)
        return inside_points_mask
    
    @property
    def features(self):
        return self.level_voxel_grid[0].features
    
    @property
    def _features(self):
        return self.level_voxel_grid[0]._features
