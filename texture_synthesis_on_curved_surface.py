import os
import cv2
import torch
import timeit
import pymesh
import xatlas
import imageio
import trimesh
import skimage
import numpy as np
import open3d as o3d
from tqdm import tqdm
from RayTracer import RayTracer
from shape_tools import write_ply, write_ply_rgb, remesh, CoACD
from skimage.util.shape import view_as_windows
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from tools.map_bvh import BvhMeshProjector
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')


def write_obj(sv_path, sv_name, mesh, img_name):
    # save obj (v, vt, f /)
    obj_file = os.path.join(sv_path, f'{sv_name}.obj')
    mtl_file = os.path.join(sv_path, f'{sv_name}.mtl')

    # print(f'[INFO] writing obj mesh to {obj_file}')
    with open(obj_file, "w") as fp:

        fp.write(f'mtllib {sv_name}.mtl \n')
        
        # print(f'[INFO] writing vertices {mesh.vertices.shape}')
        for v in mesh.vertices:
            fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
    
        # print(f'[INFO] writing vertices texture coords {mesh.visual.uv.shape}')
        for v in mesh.visual.uv:
            fp.write(f'vt {v[0]} {1 - v[1]} \n') 

        # print(f'[INFO] writing faces {mesh.faces.shape}')
        fp.write(f'usemtl defaultMat \n')
        for i in range(len(mesh.faces)):
            fp.write(f"f {mesh.faces[i, 0] + 1}/{mesh.faces[i, 0] + 1} {mesh.faces[i, 1] + 1}/{mesh.faces[i, 1] + 1} {mesh.faces[i, 2] + 1}/{mesh.faces[i, 2] + 1} \n")

    with open(mtl_file, "w") as fp:
        fp.write(f'newmtl defaultMat \n')
        fp.write(f'Ka 1 1 1 \n')
        fp.write(f'Kd 1 1 1 \n')
        fp.write(f'Ks 0 0 0 \n')
        fp.write(f'Tr 1 \n')
        fp.write(f'illum 1 \n')
        fp.write(f'Ns 0 \n')
        fp.write(f'map_Kd {img_name} \n')


def uv2vert(mesh):
    res = 2048
    device = torch.device('cuda:0')
    uvs2verts = torch.zeros([res, res, 3], dtype=torch.float32).reshape([-1, 3]).to(device)
    mask = torch.zeros([res, res, 1], dtype=torch.bool).reshape([-1]).to(device)
    vert_id = torch.arange(0, res**2).to(device)
    # Simple Mesh Projector
    meshprojector_imported = BvhMeshProjector(device=device, mesh=mesh, compute_normals=True, ini_raytracer=True, store_f=True, store_uv=True)
    # Plane Atlas Grid
    from copy import deepcopy
    mesh_plane = deepcopy(meshprojector_imported.mesh)
    mesh_plane.vertices = np.zeros_like(mesh_plane.vertices)
    mesh_plane.vertices[:, :2] = meshprojector_imported.uvs.cpu().numpy()
    meshprojector_plane = BvhMeshProjector(device=device, mesh=mesh_plane, store_f=True, compute_normals=True, ini_raytracer=True)
    # Traverse UV map
    us, vs = torch.meshgrid(torch.linspace(-1, 1, res), torch.linspace(-1, 1, res), indexing='xy')
    uvs = torch.stack([us, vs, torch.zeros_like(us)], dim=-1).reshape([-1, 3]).to(device)
    batch, start = 2048, 0
    while start < uvs.shape[0]:
        end = min(start + batch, uvs.shape[0])
        sdf, fids, barycentric = meshprojector_plane.bvh.signed_distance(uvs[start: end], return_uvw=True, mode='raystab')
        verts_3d = meshprojector_plane.barycentric_weighting(vert_values=meshprojector_imported.mesh_vertices, fids=fids, barycentric=barycentric)
        # Check uvs2verts
        uvs2verts[start: end] = verts_3d
        mask[start: end] = sdf < 1e-2
        start = end
    pcl = uvs2verts[mask].cpu().numpy()
    vert_id = vert_id[mask].cpu().numpy()
    meshprojector_imported.mesh.visual.uv = meshprojector_imported.original_uvs
    return meshprojector_imported.mesh, pcl, vert_id, res


def MeshUnion_manifold(mesh_path, sv_path):
    print('Calculate the unoin manifold of watertight parts of ' + mesh_path)
    ori_face_num = trimesh.load_mesh(mesh_path).faces.shape[0]
    mesh_name = mesh_path.split('/')[-1].split('.')[0]
    save_path = sv_path + '/' + mesh_name + '_mf.obj'
    cmd = './tools/manifold ' + f' {mesh_path} {save_path}'
    os.system(cmd)
    cmd = './tools/simplify ' + f'-i {save_path} {save_path} -f {int(.5*ori_face_num)}'
    os.system(cmd)
    print('Done with union manifold calculation! Saved in ' + save_path)
    return save_path


def Smooth(mesh_path, sv_path):
    print('Smoothing ...')
    mesh_name = mesh_path.split('/')[-1].split('.')[0]
    save_path = sv_path + '/' + mesh_name + '_sm.obj'
    new_mesh = trimesh.smoothing.filter_laplacian(trimesh.load_mesh(mesh_path), iterations=8)
    new_mesh.export(save_path)
    return save_path


def transform(data, pca, bounds, out_dim=3):
    x = data.reshape([-1, data.shape[-1]])
    x_pca = pca.transform(x)[..., :out_dim]
    x_bd = (x_pca - bounds[0]) / (bounds[1] - bounds[0])
    x_bd = np.clip(x_bd, 0., 1.)
    x_bd = x_bd.reshape([*data.shape[:-1], x_bd.shape[-1]])
    return x_bd


def get_transform(data, in_dim=None, out_dim=3):
    in_dim = data.shape[-1] if in_dim is None else in_dim
    x = data.reshape([-1, data.shape[-1]])[..., :in_dim]
    pca = PCA(n_components=out_dim)
    pca.fit(x)
    x_pca = pca.transform(x)[..., :out_dim]
    bounds = np.stack([x_pca.min(axis=0), x_pca.max(axis=0)])
    trans_func = lambda a: transform(a[..., :in_dim], pca=pca, bounds=bounds, out_dim=out_dim)
    return trans_func


class MatchingLib:
    def __init__(self, patches, channel_pca_dim=4, pyramid_height=3, pyramid_num_factor=4, pyramid_size_factor=4, quantize=True):
        self.channel_pca_dim = channel_pca_dim
        self.pyramid_height = pyramid_height
        self.pyramid_num_factor = pyramid_num_factor
        self.pyramid_size_factor = pyramid_size_factor
        self.quantize = quantize

        if self.channel_pca_dim is not None:
            self.channel_compress_func = get_transform(patches, out_dim=self.channel_pca_dim)
            patches = self.channel_compress_func(patches)
        self.patches = [patches]
        patch_num, patch_size = patches.shape[:2]
        patch_size = patches.shape[1]
        self.pyramid_sizes = [patch_size]
        self.pyramid_nums = [patch_num]
        print('Building pyramid...')
        for _ in tqdm(range(self.pyramid_height-1)):
            psize = max(4, int(self.pyramid_sizes[0] / self.pyramid_size_factor))
            pnum = max(1, int(self.pyramid_nums[-1] / self.pyramid_num_factor))
            ppatches = skimage.transform.resize(self.patches[0], (self.patches[0].shape[0], psize, psize, self.patches[0].shape[-1]))
            self.pyramid_sizes = [psize] + self.pyramid_sizes
            self.pyramid_nums.append(pnum)
            self.patches = [ppatches] + self.patches
        self.pyramid_nums = self.pyramid_nums[1:] + [1]
        
        if self.quantize:
            bounds = self.patches[-1].min(), self.patches[-1].max()
            self.bounds = bounds
            for i in range(self.pyramid_height):
                self.patches[i] = np.array((self.patches[i] - bounds[0]) / (bounds[1] - bounds[0]) * 255, dtype=np.uint8)
    
    def match(self, condition, mask):
        if self.channel_pca_dim is not None:
            condition = self.channel_compress_func(condition)
        conditions = [condition]
        masks = [mask]
        for i in range(1, self.pyramid_height):
            psize = self.pyramid_sizes[-i-1]
            pcondition = skimage.transform.resize(conditions[0], (psize, psize))
            conditions = [pcondition] + conditions
            masks = [np.array(skimage.transform.resize(masks[0], (psize, psize)) > 0)] + masks
        
        indices = np.arange(self.patches[0].shape[0])
        for i in range(self.pyramid_height):
            pcondition = conditions[i]
            if self.quantize:
                pcondition = np.array((pcondition - self.bounds[0]) / (self.bounds[1] - self.bounds[0]) * 255, dtype=np.uint8)
            error = (((pcondition[None] - self.patches[i][indices]) * masks[i][None]) ** 2).reshape(indices.shape[0], -1).sum(axis=-1)
            pindices = np.argpartition(error, self.pyramid_nums[i])[:self.pyramid_nums[i]]
            indices = indices[pindices]
        index = indices[0]
        return index


class SparseProxyDist:
    def __init__(self, dense_verts, sparse_verts=None, preferred_patch_gap=None):
        self.dense_verts = dense_verts
        self.dense_num = self.dense_verts.shape[0]
        if sparse_verts is None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(dense_verts)
            voxel_size = preferred_patch_gap / 10 if preferred_patch_gap is not None else 0.05 * (dense_verts.max() - dense_verts.min())
            downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            sparse_verts = np.asarray(downpcd.points)
        tree = KDTree(sparse_verts)
        _, index = tree.query(dense_verts, 1)
        self.d2s_index = index[..., 0]
        self.dist = np.linalg.norm(sparse_verts[:, None] - sparse_verts[None], axis=-1)
        self.sparse_verts = sparse_verts
        nn_dists, _ = tree.query(sparse_verts, 2)
        self.sparse_avg_dist = nn_dists[:, 1].mean() * 1.2
        self.arange_sparse = np.arange(self.sparse_verts.shape[0])
        # Inverse mapping
        most_mapping_num = np.bincount(self.d2s_index).max()
        self.s2d_index = self.dense_num * np.ones([self.sparse_verts.shape[0], most_mapping_num], dtype=np.int32)
        print('Sparse to dense mapping...')
        s2d_counts = np.zeros([self.sparse_verts.shape[0]], dtype=np.int32)
        for i in tqdm(range(self.dense_num)):
            sparse_idx = self.d2s_index[i]
            self.s2d_index[sparse_idx, s2d_counts[sparse_idx]] = i
            s2d_counts[sparse_idx] += 1
        
    def __getitem__(self, index):
        assert len(index) < 3, f'Shape Error! The shape of {index} is not [2]!'
        if len(index) == 1:
            index = index, slice(0, self.dense_num, None)
        sindex_x, sindex_y = self.d2s_index[index[0]], self.d2s_index[index[1]]
        return self.dist[sindex_x[:, None], sindex_y[None, :]]
    
    def range_vote(self, history_idx, mask):
        history_idx_sparse = np.unique(self.d2s_index[history_idx])
        if mask.shape[0] == self.dense_num:
            mask = np.concatenate([mask, np.zeros_like(mask[:1])], axis=0)
        not_mask = np.logical_not(mask)
        synthesized_idx = np.where(not_mask[:-1])[0]
        mask_sparse = mask[self.s2d_index].any(axis=-1)
        vert_ids_votes = (self.dist[self.d2s_index[not_mask[:-1], None], history_idx_sparse[None, :]] * mask_sparse[history_idx_sparse]).sum(axis=1)
        next_syn_vert_id = synthesized_idx[np.argmax(vert_ids_votes)]
        return next_syn_vert_id
    
    def pick_vertices_to_set(self, tree_verts, grid_gap):
        tree = KDTree(tree_verts)
        # Use sparse proxy to filter points
        nn_dist_sparse, _ = tree.query(self.sparse_verts, 1)
        nn_dist_sparse = nn_dist_sparse[..., 0]
        filtered_sparse_mask = nn_dist_sparse < self.sparse_avg_dist * 2
        filtered_sparse_idx = self.arange_sparse[filtered_sparse_mask]
        filtered_dense_mask = np.in1d(self.d2s_index, filtered_sparse_idx)
        filtered_dense_idx = np.where(filtered_dense_mask)[0]
        # Use filtered dense points to pick vertices to set
        nn_dist, _ = tree.query(verts[filtered_dense_mask], 1)
        nn_dist = nn_dist[..., 0]
        nn_dist_mask = nn_dist < grid_gap
        verts_to_set_id = filtered_dense_idx[nn_dist_mask]
        return verts_to_set_id


def prepareExamplePatches(exemplar, patchSize, overlapSize=None, windowStep=5, mirror_hor=True, mirror_vert=True):
    overlapSize = int(patchSize / 4) if overlapSize is None else overlapSize
    searchKernelSize = patchSize + 2 * overlapSize
    result = view_as_windows(exemplar, [searchKernelSize, searchKernelSize, 3] , windowStep)
    shape = np.shape(result)
    result = result.reshape(shape[0]*shape[1], searchKernelSize, searchKernelSize, 3)
    total_patches_count = shape[0]*shape[1]
    if mirror_hor:
        hor_result = np.zeros(np.shape(result))
        for i in range(total_patches_count):
            hor_result[i] = result[i][::-1, :, :]
        result = np.concatenate((result, hor_result))
    if mirror_vert:
        vert_result = np.zeros((shape[0]*shape[1], searchKernelSize, searchKernelSize, 3))
        for i in range(total_patches_count):
            vert_result[i] = result[i][:, ::-1, :]
        result = np.concatenate((result, vert_result))
    return result


def map_uv(mesh):
    if hasattr(mesh.visual, 'uv'):
        print('Use original UV')
        uvs = mesh.visual.uv
    else:
        print('Use xatlas UV mapping')
        vmapping, faces, uvs = xatlas.parameterize(mesh.vertices, mesh.faces)
        mesh = trimesh.Trimesh(vertices=mesh.vertices[vmapping], faces=faces, process=False)
        uvs = (uvs - uvs.min()) / (uvs.max() - uvs.min()) * 2 - 1
    return mesh, uvs


def define_vector_field(mesh):
    mesh.as_open3d.compute_vertex_normals()
    default_vector = np.array([0., 1., 0.], dtype=np.float32)
    normals = mesh.as_open3d.vertex_normals
    vectors = default_vector - (normals * default_vector).sum(axis=-1, keepdims=True) * normals
    return vectors


def points_to_barycentric(triangles, points):
    points = points[..., None,:]
    p2v = triangles - points
    s0 = np.linalg.norm(np.cross(p2v[..., 1, :], p2v[..., 2, :], axis=-1), axis=-1)
    s1 = np.linalg.norm(np.cross(p2v[..., 2, :], p2v[..., 0, :], axis=-1), axis=-1)
    s2 = np.linalg.norm(np.cross(p2v[..., 0, :], p2v[..., 1, :], axis=-1), axis=-1)
    barycentric = np.stack([s0, s1, s2], axis=-1)
    barycentric = barycentric / (barycentric.sum(axis=-1, keepdims=True) + 1e-8)
    return barycentric


def extract_patch_on_surface(meshprojector, vert, patchSize, vectors, grid_gap):
    calibration = np.linspace(-patchSize*grid_gap/2, patchSize*grid_gap/2, patchSize)
    x, y = np.meshgrid(calibration, calibration, indexing='ij')
    patch_coor = np.stack([x, y], axis=-1).reshape([-1, 2])
    patch_coor = np.concatenate([patch_coor, np.zeros_like(patch_coor)], axis=-1)
    patch_coor[..., -1] = 1

    patch_vert_ids = np.arange(patch_coor.shape[0]).reshape([patchSize, patchSize])
    patch_faces = []
    for i in range(patchSize - 1):
        for j in range(patchSize - 1):
            patch_faces.append([patch_vert_ids[i, j], patch_vert_ids[i+1, j], patch_vert_ids[i, j+1]])
            patch_faces.append([patch_vert_ids[i+1, j], patch_vert_ids[i+1, j+1], patch_vert_ids[i, j+1]])
    patch_faces = np.stack(patch_faces)

    # Determine the transform matrix by sample vertex
    nn_vertex_id, _, _, _, _ = meshprojector.barycentric_mapping(torch.from_numpy(vert[None]).float().cuda(), return_face_id=True)
    nn_vertex_id = nn_vertex_id[..., 0]
    normals = meshprojector.mesh.as_open3d.vertex_normals
    z_axis = normals[nn_vertex_id]
    y_axis = np.cross(z_axis, vectors[nn_vertex_id])
    if (y_axis == 0).all():
        y_axis = np.cross(z_axis, np.array([1., 1., 1.01]) + vectors[nn_vertex_id])
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    T = np.eye(4)
    T[:3, :3] = np.stack([x_axis, y_axis, z_axis], -1)
    T[:3, 3] = vert

    ray_origins = np.einsum('ab,nb->na', T, patch_coor)[..., :3]
    shooting_distance = 0.05
    # Ray casting
    device = torch.device("cuda:0")
    ray_origins = torch.from_numpy(ray_origins).float().to(device)
    ray_origins += shooting_distance * torch.from_numpy(z_axis).float().to(device)
    ray_directions = torch.from_numpy(np.broadcast_to(-z_axis[None], ray_origins.shape)).float().to(device)
    raytracer = RayTracer(mesh.vertices, mesh.faces)
    intersections, _, depth, inter_faces = raytracer.trace(ray_origins, ray_directions)
    inter_faces = inter_faces.reshape((patchSize, patchSize)).cpu().numpy()

    # Ray cast check
    depth_np = depth.cpu().numpy().reshape((patchSize, patchSize))
    mask = depth_np < 9.5
    # Normal angle check
    inter_faces_normals = np.asarray(mesh.as_open3d.triangle_normals)[inter_faces.reshape([-1])].reshape([*inter_faces.shape, 3])
    normal_check_mask = (inter_faces_normals * z_axis).sum(axis=-1) > np.cos(np.pi / 4)
    mask = np.logical_and(mask, normal_check_mask)
    # Depth check
    depth_check_mask = np.abs(depth_np - shooting_distance) < 0.05
    mask = np.logical_and(mask, depth_check_mask)
    # Remove isolations and holes
    kernel = np.ones((3, 3), dtype=float)
    mask = cv2.erode(np.array(mask, dtype=float)[..., None], kernel=kernel, iterations=2)
    mask = cv2.dilate(np.array(mask, dtype=float)[..., None], kernel=kernel, iterations=2)
    mask = cv2.dilate(np.array(mask, dtype=float)[..., None], kernel=kernel, iterations=2)
    mask = cv2.erode(np.array(mask, dtype=float)[..., None], kernel=kernel, iterations=2)
    mask = mask > 0

    uvh, _, _, _ = meshprojector.uvh(intersections)
    intersections_uvs = uvh[..., :2]
    intersections = intersections.reshape([patchSize, patchSize, 3]).cpu().numpy()
    return intersections, intersections_uvs, mask, patch_faces


def synthesis_on_uvmap(mesh, verts, vert_ids, resolution, patches, vectors, original_grid_gap, grid_gap=3e-4, sv_path=None, use_matchlib=True, measure_time=True, range_voting=True):
    sv_path = './test_data_nerf/' if sv_path is None else sv_path
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
    
    trans_func = get_transform(patches, 3)
    textures = torch.zeros((1, patches.shape[-1], resolution, resolution), dtype=torch.float32).cuda()
    syn_mask = torch.zeros((1, 1, resolution, resolution), dtype=torch.float32).cuda()
    syn_mask1d = np.zeros((verts.shape[0]), dtype=np.int8)
    meshprojector = BvhMeshProjector(device=torch.device('cuda:0'), mesh=mesh, compute_normals=True, ini_raytracer=True, store_f=True, store_uv=True)

    last_syn_vert_id = -1
    history_syn_vert_ids = [0]
    patchSize = patches.shape[1]
    preferred_patch_gap = patchSize * grid_gap * .9
    print('Preferred patch gap: ', preferred_patch_gap)

    # Matching Library
    if use_matchlib:
        malib = MatchingLib(patches=patches, channel_pca_dim=None, pyramid_height=2, pyramid_num_factor=10, pyramid_size_factor=8, quantize=False)

    # Vertices Distance
    sparse_verts_proxy = SparseProxyDist(verts, mesh.vertices, preferred_patch_gap)
    if range_voting:
        vert_in_range = sparse_verts_proxy
        vert_in_range.dist = np.logical_and(vert_in_range.dist < preferred_patch_gap * 1., vert_in_range.dist > preferred_patch_gap * .8) - (vert_in_range.dist <= preferred_patch_gap * .8) * 1.
    
    while not syn_mask1d.all():
        if measure_time:
            times = [timeit.default_timer()]
            checkpoints = []
        #############################################################################################################################
        #############################################################################################################################

        if range_voting:
            next_syn_vert_id = sparse_verts_proxy.range_vote(history_idx=history_syn_vert_ids, mask=syn_mask1d)
        else:
            # Preferred distance history
            vert_dist_to_history_syn = sparse_verts_proxy[:, history_syn_vert_ids].min(axis=1)
            vert_ids_dist_to_history_syn = np.abs(vert_dist_to_history_syn - preferred_patch_gap) + 1e5 * syn_mask1d
            next_syn_vert_id = np.argmin(vert_ids_dist_to_history_syn)
        
        # Loop check
        log_check_flag = False
        if next_syn_vert_id == last_syn_vert_id:
            print('Potential be a dead loop! Set log_check_flag to True!')
            log_check_flag = True
            # import pdb
            # pdb.set_trace()
        last_syn_vert_id = next_syn_vert_id
        history_syn_vert_ids.append(last_syn_vert_id)

        if measure_time:
            times.append(timeit.default_timer())
            checkpoints.append('1.Point picking')
        #############################################################################################################################
        #############################################################################################################################
        
        # Extract the patch from the surface
        patch_verts, intersections_uvs, patch_mask, patch_faces = extract_patch_on_surface(meshprojector, vert=verts[next_syn_vert_id], patchSize=patchSize, vectors=vectors, grid_gap=grid_gap)
        occupied_mask = (torch.nn.functional.grid_sample(syn_mask, intersections_uvs[None, None], align_corners=True, padding_mode="zeros").squeeze().cpu().numpy().reshape([patchSize, patchSize]) > 0.9) * patch_mask
        synthesized_parts = torch.nn.functional.grid_sample(textures, intersections_uvs[None, None, :], align_corners=True, padding_mode="zeros").squeeze().permute(1, 0).cpu().numpy().reshape([patchSize, patchSize, -1])
        
        if measure_time:
            times.append(timeit.default_timer())
            checkpoints.append('2.Patch extraction')
        #############################################################################################################################
        #############################################################################################################################

        # Blend the border
        kernel = np.ones((3, 3), dtype=np.float32)
        smooth_range = int(patchSize/20)
        blend_masks = [np.array(occupied_mask, dtype=np.float32)]
        for _ in range(smooth_range):
            blend_masks.append(cv2.erode(blend_masks[-1][..., None], kernel=kernel, iterations=1))
        blend_mask = np.stack(blend_masks, axis=0).mean(axis=0)[..., None]
        match_mask = occupied_mask[..., None] - blend_mask
        
        if measure_time:
            times.append(timeit.default_timer())
            checkpoints.append('3.Border blending')
        #############################################################################################################################
        #############################################################################################################################
        
        # Calculate error and patch match
        if use_matchlib:
            picked_patch_id = malib.match(synthesized_parts, match_mask)
            picked_patch = patches[picked_patch_id]
        else:
            error = (((patches - synthesized_parts[None]) ** 2) * match_mask[None]).reshape([patches.shape[0], -1]).sum(axis=-1)
            picked_patch_id = error.argmin()
            picked_patch = patches[picked_patch_id]
        # Patch quilting
        picked_patch = picked_patch * (1 - blend_mask) + synthesized_parts * blend_mask
        
        if measure_time:
            times.append(timeit.default_timer())
            checkpoints.append('4.Matching and quilting')
        #############################################################################################################################
        #############################################################################################################################

        # Picking vertices close to patch grid to set textures
        erode_occupied_mask = cv2.erode(np.array(occupied_mask, dtype=float)[..., None], kernel=kernel, iterations=1) > 0
        tree_verts = patch_verts[2:-2, 2:-2].reshape([-1, 3])[np.logical_and(np.logical_not(erode_occupied_mask), patch_mask)[2:-2, 2:-2].reshape([-1])]  # Only set inside and not occupied
        if len(tree_verts) == 0:
            tree_verts = np.array([verts[next_syn_vert_id]])
        verts_to_set_id = sparse_verts_proxy.pick_vertices_to_set(tree_verts=tree_verts, grid_gap=grid_gap)
        verts_to_set_id = np.union1d(verts_to_set_id, np.array([next_syn_vert_id]))  # Force setting selected points!
        
        if measure_time:
            times.append(timeit.default_timer())
            checkpoints.append('5.Picking vertices to set')
        #############################################################################################################################
        #############################################################################################################################

        # Set vertices texture with barycentrics on patch_grid
        patch_mesh = trimesh.Trimesh(patch_verts.reshape([-1, 3]), patch_faces)
        _, dist, v2p_face_id = trimesh.proximity.closest_point(patch_mesh, verts[verts_to_set_id])
        finer_mask = dist < 1e-3 if not log_check_flag else dist < np.inf
        verts_to_set_id, v2p_face_id = verts_to_set_id[finer_mask], v2p_face_id[finer_mask]
        barycentric_mesh_verts = points_to_barycentric(patch_mesh.vertices[patch_mesh.faces[v2p_face_id]], verts[verts_to_set_id])
        synthesizing_textures = (picked_patch.reshape([-1, picked_patch.shape[-1]])[patch_mesh.faces[v2p_face_id]] * barycentric_mesh_verts[..., None]).sum(axis=-2)
        textures[0, :, vert_ids[verts_to_set_id] // resolution, vert_ids[verts_to_set_id] % resolution] = torch.from_numpy(synthesizing_textures).to(textures.device).float().permute(1, 0)
        syn_mask[0, :, vert_ids[verts_to_set_id] // resolution, vert_ids[verts_to_set_id] % resolution] = 1
        syn_mask1d[verts_to_set_id] = 1
        
        if measure_time:
            times.append(timeit.default_timer())
            checkpoints.append('6.Texture setting')
        #############################################################################################################################
        #############################################################################################################################

        # Visualize
        patch_mesh = trimesh.Trimesh(patch_verts.reshape([-1, 3]), patch_faces)
        patch_mesh.visual.vertex_colors = trans_func(picked_patch).reshape([-1, 3])
        patch_mesh.export(sv_path + '/patch_mesh.obj')
        write_ply_rgb(patch_verts.reshape([-1, 3]), np.array(trans_func(picked_patch).reshape([-1, 3]) * 255), './test_data_nerf/patch_verts.ply')
        img = trans_func(textures[0].permute(1, 2, 0).cpu().numpy())
        Image.fromarray(np.array(img * 255, dtype=np.uint8)).save(sv_path + '/output.png')
        mesh = trimesh.Trimesh(vertices=meshprojector.mesh.vertices, faces=meshprojector.mesh.faces, process=False)
        mesh.visual.uv = meshprojector.original_uvs
        write_obj(sv_path, 'output', mesh, 'output.png')
        
        if measure_time:
            times.append(timeit.default_timer())
            checkpoints.append('7.Visualization')
        #############################################################################################################################
        #############################################################################################################################

        if measure_time:
            times = np.array(times)
            times = times[1:] - times[:-1]
            print('Process: %.2f' % (syn_mask1d.sum() / syn_mask1d.shape[0] * 100), '% ', end='')
            for i in range(times.shape[0]):
                print(checkpoints[i], ': %.2f' % times[i], 's. ', end='')
            print(', ', syn_mask1d.shape[0] - syn_mask1d.sum().item(), ' points left.')
            left = np.arange(times.shape[0])
            plt.figure(figsize=(50, 10))
            plt.bar(left, times, linewidth=2, tick_label=checkpoints)
            plt.xticks(fontsize=12)
            plt.show()
            plt.savefig(sv_path + '/time_measurement.png')
            plt.clf()
            plt.close()
        else:
            print('Process: %.2f' % (syn_mask1d.sum() / syn_mask1d.shape[0] * 100), '% ', syn_mask1d.shape[0] - syn_mask1d.sum().item(), ' points left.')

        plt.subplot(231)
        plt.gca().set_title('Synthesized region')
        plt.imshow(trans_func(synthesized_parts) * occupied_mask[..., None])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(232)
        plt.gca().set_title('Blend Mask')
        plt.imshow(blend_mask)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(233)
        plt.gca().set_title('Picked Patch')
        plt.imshow(trans_func(patches[picked_patch_id]))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(234)
        plt.gca().set_title('Blended Patch')
        plt.imshow(trans_func(picked_patch))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(235)
        plt.gca().set_title('Patch Mask')
        plt.imshow(patch_mask)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(236)
        plt.gca().set_title('Occupied Mask')
        plt.imshow(occupied_mask)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(sv_path + '/visualization.png')
        plt.clf()
        plt.close()
    
    mesh.visual.vertex_colors = trans_func(textures.cpu()[0].permute(1, 2, 0).numpy())
    mesh.export(sv_path + '/result.obj')
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    np.savez(sv_path + '/curved_mesh.npz', features=textures.cpu().numpy(), mesh=mesh, uv=meshprojector.original_uvs, phi_embed=None, local_tbn=None, sdf_factor=grid_gap / original_grid_gap, original_grid_gap=original_grid_gap)


if __name__ == '__main__':
    dataset_name = 'wall'
    curved_surface_name = 'bunny'

    # model_name = 'curved_grid_hash_clus_optcam_SH'
    # model_name = 'curved_grid_hash_clus_optcam_None'
    # model_name = 'curved_grid_hash_clus_SH_SM'
    model_name = 'curved_grid_hash_clus_Ref'
    # model_name = 'curved_grid_hash_clus_None'
    # model_name = 'curved_grid_hash_clus_optcam_None_novis'
    # model_name = 'curved_grid_hash_optcam_SH_novis'
    # model_name = 'curved_grid_hash_clus_optcam_None'

    grid_gap = 5e-4
    data_path = f'PATH/TO/LOG/{dataset_name}/field/'
    data = np.load(f'{data_path}/{model_name}.npz', allow_pickle=True)
    patches = data['patches']
    sv_path = f'PATH/TO/LOG/{dataset_name}/field/curved_surface/' + dataset_name + '/' + curved_surface_name + '/'
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)

    mirror_hor = True
    mirror_ver = True
    crop_shift_augmentation = True

    if mirror_hor:
        patches_hor = patches[:, ::-1]
        patches = np.concatenate([patches, patches_hor], axis=0)
    if mirror_ver:
        patches_ver = patches[:, :, ::-1]
        patches = np.concatenate([patches, patches_ver], axis=0)

    crop_out_len = int(patches.shape[1] // 5)
    crop_len = patches.shape[1] - crop_out_len
    crop_aug_fator = 2
    if crop_shift_augmentation:
        stride = crop_out_len // crop_aug_fator
        offset = np.arange(crop_aug_fator) * stride
        patches_crop = patches[:, :crop_len, :crop_len]
        for i in range(crop_aug_fator):
            for j in range(crop_aug_fator):
                if i==0 and j==0:
                    continue
                patches_crop = np.concatenate([patches_crop, patches[:, offset[i]: crop_len+offset[i], offset[j]: crop_len+offset[j]]])
    
    print('Total patch number: ', patches.shape[0])

    patchSize = patches.shape[1]
    simple_mesh_path = f'PATH/TO/MESH/{curved_surface_name}.obj'
    if not os.path.exists(sv_path + f'/{curved_surface_name}_mf_sm.obj'):
        print('No such file: ', sv_path + f'/{curved_surface_name}_mf_sm.obj', ' Generating...')
        simple_mesh_path = CoACD(mesh_path=simple_mesh_path, threshold=0.05)
        simple_mesh_path = MeshUnion_manifold(simple_mesh_path, sv_path)
        mesh = pymesh.load_mesh(simple_mesh_path)
        mesh = remesh(mesh, "low")
        pymesh.save_mesh(simple_mesh_path, mesh)
        simple_mesh_path = Smooth(simple_mesh_path, sv_path)
    else:
        print(sv_path + f'/{curved_surface_name}_mf_sm.obj', ' already existed.')
        simple_mesh_path = sv_path + f'/{curved_surface_name}_mf_sm.obj'
    mesh = trimesh.load_mesh(simple_mesh_path)
    mesh.vertices -= mesh.vertices.mean(axis=0)
    mesh.vertices /= (1.5 * np.abs(mesh.vertices).max())
    mesh, verts, vert_ids, res = uv2vert(mesh)
    vectors = define_vector_field(mesh)
    synthesis_on_uvmap(mesh, verts, vert_ids, res, grid_gap=grid_gap, patches=patches, vectors=vectors, original_grid_gap=data['grid_gap'], sv_path=sv_path)
