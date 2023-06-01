import os
import frnn
import torch
import xatlas
import trimesh
import pytorch3d
import matplotlib
import numpy as np
import open3d as o3d
import torch.nn as nn
from tqdm import tqdm
from pytorch3d import _C
import tinycudann as tcnn
import torch.nn.functional as F
from RayTracer import RayTracer
import matplotlib.pyplot as plt
from tools.encoding import get_encoder
from pytorch3d.io import load_obj
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from pytorch3d.structures import Meshes, Pointclouds
from gridencoder.grid_clustering import ClusteringLayer
from tools.shape_tools import write_ply


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def batchify(fn, chunk=1024*32, axis=0):
    """Render rays in smaller minibatches to avoid OOM.
    """
    if chunk is None:
        return fn

    def ret_func(**kwargs):
        x = kwargs[list(kwargs.keys())[0]]
        all_ret = {}
        for i in range(0, x.shape[axis], chunk):
            end = min(i + chunk, x.shape[axis])
            chunk_kwargs = dict([[key, torch.index_select(kwargs[key], axis, torch.arange(i, end).to(kwargs[key].device))] for key in kwargs.keys()])
            ret = fn(**chunk_kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], axis) for k in all_ret}
        return all_ret

    return ret_func


def load_model(mesh_path, device=device):
    verts, faces, aux = pytorch3d.io.load_obj(mesh_path, device=device)
    faces_idx = faces.verts_idx.to(device)
    faces_t = faces.textures_idx.to(device)
    verts_uvs = aux[1].to(device) if aux[1] is not None else None
    return verts, faces, faces_idx, verts_uvs, faces_t, device


def point_mesh_face_distance(meshes: Meshes, pcls: Pointclouds, min_triangle_area=5e-3):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    point_to_face, idxs = _C.point_face_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area
    )
    
    return point_to_face, idxs


def points_to_barycentric(triangles, points):
    points = points.unsqueeze(-2)
    p2v = triangles - points
    s0 = torch.cross(p2v[..., 1, :], p2v[..., 2, :], dim=-1).norm(dim=-1)
    s1 = torch.cross(p2v[..., 2, :], p2v[..., 0, :], dim=-1).norm(dim=-1)
    s2 = torch.cross(p2v[..., 0, :], p2v[..., 1, :], dim=-1).norm(dim=-1)
    barycentric = torch.stack([s0, s1, s2], dim=-1)
    barycentric = barycentric / (barycentric.sum(dim=-1, keepdim=True) + 1e-5)
    return barycentric


def diagonal_dot(a, b):
    return torch.matmul(a * b, torch.ones(a.shape[1]).to(a.device))


def barycentric_to_points(triangles, barycentric):
    return (triangles * barycentric.view((-1, 3, 1))).sum(dim=1)


def p2f_dist_bachify(pfc_verts, pfc_faces, points, batch_size=64*64):
    start = 0
    idx = torch.zeros([pfc_faces.shape[0]], dtype=torch.long, device=pfc_verts.device)
    while start < pfc_verts.shape[0]:
        end = min(start+batch_size, pfc_verts.shape[0])
        meshes = Meshes(verts=pfc_verts[start: end], faces=pfc_faces[start: end])
        pcls = Pointclouds(points=points[0, start: end].unsqueeze(1))
        _, idx_ = point_mesh_face_distance(meshes, pcls)
        offset = torch.linspace(0, (idx_.shape[0]-1)*pfc_faces.shape[1],  idx_.shape[0], dtype=torch.long, device=idx_.device)
        idx_ = idx_ - offset
        idx[start: end] = idx_
        start = end
    return idx


def calculate_tbn(mesh, uvs, force_orthogonal=True):
    # Calculate TBN coordinate system from mesh and uvs (N, 2)
    # https://learnopengl.com/Advanced-Lighting/Normal-Mapping
    vertices, faces, normals = mesh.vertices, mesh.faces, mesh.face_normals
    faces_verts = vertices[faces]  # M, 3, 3
    faces_uvs = uvs[faces]  # M, 3, 2
    faces_verts_edges = faces_verts[:, 1:] - faces_verts[:, :1]  # M, 2, 3
    faces_uvs_edges = faces_uvs[:, 1:] - faces_uvs[:, :1]  # M, 2, 2
    # Check singular issue
    rank = np.linalg.matrix_rank(faces_uvs_edges)
    if rank.min() < 2:
        idx = np.where(rank < 2)[0]
        faces_uvs_edges[idx, 1, 1] += 1e-3
    # Calculate tbn
    faces_tb = np.einsum('mab,mbc->mac', np.linalg.inv(faces_uvs_edges), faces_verts_edges)  # M, 2, 3
    faces_tbn = np.concatenate([faces_tb, normals[:, None]], axis=1)  # M, 3, 3
    if force_orthogonal:
        faces_tbn[:, 1] = np.cross(faces_tbn[:, 2], faces_tbn[:, 0], axis=-1)
    faces_tbn = faces_tbn / np.linalg.norm(faces_tbn, axis=-1, keepdims=True)
    return faces_tbn


class project_layer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xyz, knn_func, trace_func, K, depth_threshold, h_threshold):
        # Decide the projection direction
        normal, _, _, _ = knn_func(xyz=xyz, K=K)
        # Project to mesh surface
        p_sur_1, _, depth_1, _ = trace_func(xyz, normal)  # inner
        p_sur_2, _, depth_2, _ = trace_func(xyz, -normal)  # outer
        condition = depth_1 < depth_2
        p_sur = torch.where(condition.unsqueeze(-1), p_sur_1, p_sur_2)
        sdf = torch.where(condition, -depth_1, depth_2).unsqueeze(-1)
        # Calculate the mask
        if h_threshold is None:
            h_threshold = np.inf
        h_mask = (sdf.abs() < min(depth_threshold, h_threshold)).squeeze(-1)
        # Save for backward
        ctx.save_for_backward(normal)
        return p_sur, sdf, h_mask, normal

    @staticmethod
    def backward(ctx, g_psur, g_sdf, g_hmask, g_normal):
        normal = ctx.saved_tensors[0]
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-5)
        g_xyz_parallel2surface = g_psur - normal * (normal * g_psur).sum(dim=-1, keepdim=True)
        g_xyz_along_normal = g_sdf * normal
        g_xyz = g_xyz_along_normal + g_xyz_parallel2surface
        return g_xyz, None, None, None, None, None


class diff_project_layer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xyz, p_sur, sdf, normal):
        # Save for backward
        ctx.save_for_backward(normal)
        return xyz, p_sur, sdf, normal

    @staticmethod
    def backward(ctx, g_xyz, g_psur, g_sdf, g_normal):
        normal = ctx.saved_tensors[0]
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-5)
        g_xyz_parallel2surface = g_psur - normal * (normal * g_psur).sum(dim=-1, keepdim=True)
        g_xyz_along_normal = g_sdf * normal
        g_xyz = g_xyz_along_normal + g_xyz_parallel2surface
        return g_xyz, g_psur, g_sdf, g_normal


class LipMLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_neurons=64, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        dim = in_dim
        for _ in range(num_layers):
            self.layers.append(LipLayer(dim, n_neurons))
            dim = n_neurons
        self.layers.append(LipLayer(dim, out_dim, act=nn.Identity()))
        self.layers_seq = nn.Sequential(*self.layers)
    
    def forward(self, x):
        y = self.layers_seq(x)
        return y

    def regularization(self):
        loss = 1.
        for i in range(len(self.layers)):
            loss = loss * self.layers[i].softplus(self.layers[i].c)
        return loss
    

class LipLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU()):
        super().__init__()
        self.act = act
        self.W = nn.Parameter(torch.randn(out_dim, in_dim).float() * 1e-1)
        self.b = nn.Parameter(torch.zeros(out_dim, dtype=torch.float32))
        self.c = nn.Parameter(torch.ones([], dtype=torch.float32))
        self.softplus = nn.Softplus()

    def normalization(self):
        absrowsum = self.W.abs().sum(dim=1)
        softplus_c = self.softplus(self.c)
        scale = torch.minimum(torch.ones_like(absrowsum), softplus_c / absrowsum)
        return self.W * scale[..., None]

    def forward(self, x):
        y = self.act(torch.einsum('ab,nb->na', self.normalization(), x) + self.b)
        return y


class Factorized_Normal_Net(nn.Module):
    def __init__(self, x_dim, z_dim, theta_scale=np.pi/2*1.1, phi_scale=np.pi*2*1.1, bound_output=False, lip=True, low_freq_band_len_f=32, low_freq_band_len_z=12, direct_pred_coor=False):
        super().__init__()
        # Predict phi independently, which is anisotropic
        self.encoder, self.encoder_out_dim = get_encoder("hashgrid", desired_resolution=1024, input_dim=3, num_levels=4, level_dim=2, base_resolution=512, log2_hashmap_size=19, align_corners=True)
        self.low_freq_band_len_x = min(x_dim, low_freq_band_len_f)
        self.low_freq_band_len_z = min(z_dim, low_freq_band_len_z)

        self.direct_pred_coor = direct_pred_coor

        if direct_pred_coor:
            if lip:
                self.coor_net = LipMLP(in_dim=self.encoder_out_dim+self.low_freq_band_len_z, out_dim=3, n_neurons=32, num_layers=3)
            else:
                self.coor_net = tcnn.Network(
                        n_input_dims=self.encoder_out_dim+self.low_freq_band_len_z,
                        n_output_dims=3,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "None",
                            "n_neurons": 32,
                            "n_hidden_layers": 2,
                        },
                    )
        else:
            if lip:
                self.phi_net = LipMLP(in_dim=self.encoder_out_dim+self.low_freq_band_len_z, out_dim=1, n_neurons=16, num_layers=2)
                # Predict theta based on geometry features
                self.theta_net = LipMLP(in_dim=self.low_freq_band_len_x+self.low_freq_band_len_z, out_dim=1, n_neurons=16, num_layers=2)
            else:
                self.phi_net = tcnn.Network(
                        n_input_dims=self.encoder_out_dim+self.low_freq_band_len_z,
                        n_output_dims=1,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "None",
                            "n_neurons": 16,
                            "n_hidden_layers": 2,
                        },
                    )
                self.theta_net = tcnn.Network(
                        n_input_dims=self.low_freq_band_len_x+self.low_freq_band_len_z,
                        n_output_dims=1,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "None",
                            "n_neurons": 16,
                            "n_hidden_layers": 2,
                        },
                    )
            
        self.theta_scale = theta_scale
        self.phi_scale = phi_scale
        self.sigmoid = nn.Sigmoid()
        self.bound_output = bound_output
        self.lip = lip
    
    def regularization(self):
        if self.lip:
            if self.direct_pred_coor:
                return self.coor_net.regularization()
            else:
                return self.phi_net.regularization() + self.theta_net.regularization()
        else:
            return 0.
    
    def toCoor(self, phi, theta):
        sin_theta = torch.sin(theta)
        return torch.cat([sin_theta * torch.cos(phi), sin_theta * torch.sin(phi), torch.cos(theta)], axis=-1)
    
    def rot(self, phi, theta, tbn):
        rot = self.toCoor(phi=phi, theta=theta)
        normal_fine = torch.einsum('na,nab->nb', rot, tbn)
        return normal_fine
    
    def phi_embedding(self, p_sur):
        phi_embed = self.encoder(p_sur)
        return phi_embed

    def forward(self, z_embed, x_embed, p_sur=None, phi_embed=None, tbn=None, return_rot_angles=False):
        # z_embed, phi_embed are independent of texture latents, which learn anisotropic attribute phi
        # x_embed learns isotropic attributes like theta, color, sigma
        assert p_sur is None or phi_embed is None, 'Only one of p_sur and phi_embed is None'
        if p_sur is not None:
            phi_embed = self.encoder(p_sur)

        if self.direct_pred_coor:
            normal = self.coor_net(torch.cat([phi_embed, z_embed[..., :self.low_freq_band_len_z]], dim=-1))
            normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-5)
        else:
            geo_feat = torch.cat([x_embed[..., :self.low_freq_band_len_x], z_embed[..., :self.low_freq_band_len_z]], dim=-1)
            phi = self.phi_net(torch.cat([phi_embed, z_embed[..., :self.low_freq_band_len_z]], dim=-1))
            theta = self.theta_net(geo_feat)
            if self.bound_output:
                theta = self.theta_scale * self.sigmoid(theta)
                phi = self.phi_scale * self.sigmoid(phi)
            if return_rot_angles:
                return theta, phi
            normal = self.toCoor(phi=phi, theta=theta)
        
        if tbn is None:
            return normal
        else:
            return torch.einsum('na,nab->nb', normal, tbn)


class MeshProjector:
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), mesh_path=None, mesh=None, mean_edge_length=None, distance_method="frnn", compute_normals=True, ini_raytracer=True, store_f=False, store_uv=True):
        # Initialize mesh
        if mesh is None:
            self.mesh = trimesh.load_mesh(mesh_path)
        else:
            self.mesh = mesh
        
        # UV
        if store_uv:
            if hasattr(self.mesh.visual, 'uv'):
                print('Use original uv')
                uvs = torch.FloatTensor(self.mesh.visual.uv).to(device)
            else:
                print('Use xatlas UV mapping ...')
                vmapping, faces, uvs = xatlas.parametrize(self.mesh.vertices, self.mesh.faces)
                self.mesh = trimesh.Trimesh(vertices=self.mesh.vertices[vmapping], faces=faces, process=False)
                self.mesh.visual.vertex_colors = np.zeros_like(self.mesh.vertices, dtype=np.uint8)
                self.mesh.visual.vertex_colors[:, :2] = np.array(uvs * 255, dtype=np.uint8)
                self.mesh.export('./test_data/uv_mapped.obj')
                uvs = torch.FloatTensor(uvs).to(device)
            self.uvs = (uvs - uvs.min()) / (uvs.max() - uvs.min()) * 2 - 1.  # Map to -1~1
        else:
            self.uvs = None
        
        # Calculate local Tangent, Bitangent, Normal coordinate system
        self.tbn = torch.from_numpy(calculate_tbn(self.mesh, self.uvs.cpu().numpy())).float().to(device) if self.uvs is not None else None

        # Compute normals
        if compute_normals:
            self.mesh.as_open3d.compute_vertex_normals()

        # Calculate mean edge length
        if mean_edge_length is None:
            edges = self.mesh.vertices[self.mesh.edges_unique]
            edges = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=-1)
            self.mean_edge_length = edges.mean()
        else:
            self.mean_edge_length = mean_edge_length

        # Calculate recommended sdf factor
        if self.uvs is not None:
            uvs = self.uvs.cpu().numpy()
            edges_uv = uvs[self.mesh.edges_unique]
            edges_uv = np.linalg.norm(edges_uv[:, 0] - edges_uv[:, 1], axis=-1)
            mean_edges_uv = edges_uv.mean()
            self.recommended_sdf_factor = self.mean_edge_length / mean_edges_uv
        else:
            self.recommended_sdf_factor = None

        self.gaussian_factor = - np.log(np.e) / (self.mean_edge_length ** 2)

        # Store vertices and create frnn
        self.mesh_vertices = torch.FloatTensor(self.mesh.vertices).to(device)
        print(f'Mesh Projector with {self.mesh_vertices.shape[0]} vertices')
        self.vertex_normals = torch.FloatTensor(np.asarray(self.mesh.as_open3d.vertex_normals)).to(device)
        _, _, _, self.grid = frnn.frnn_grid_points(self.mesh_vertices.unsqueeze(0), self.mesh_vertices.unsqueeze(0), None, None, K=8, r=100., grid=None, return_nn=False, return_sorted=True)
        self.radius = 100.
        self.distance_method = distance_method
        self.max_K = self.mesh_vertices.shape[0]
        
        # Initialize raytracer
        if ini_raytracer:
            self.raytracer = RayTracer(self.mesh.vertices, self.mesh.faces)
        else:
            self.raytracer = None
        self.depth_threshold = 9.5

        # Store faces
        if store_f:
            self.faces = torch.from_numpy(self.mesh.faces).to(device).long()
        else:
            self.faces = None

    def project(self, xyz, K=8, h_threshold=None, requires_grad_xyz=False, use_dir_vec=True):
        # if requires_grad_xyz:
        #     return project_layer.apply(xyz, self.knn, self.raytracer.trace, K, self.depth_threshold, h_threshold)
        # else:
        normal, _, _, _ = self.knn(xyz=xyz, K=K, use_dir_vec=use_dir_vec)
        p_sur_1, _, depth_1, face_idx_1 = self.raytracer.trace(xyz, normal)  # inner
        p_sur_2, _, depth_2, face_idx_2 = self.raytracer.trace(xyz, -normal)  # outer
        condition = depth_1 < depth_2
        p_sur = torch.where(condition.unsqueeze(-1), p_sur_1, p_sur_2)
        sdf = torch.where(condition, -depth_1, depth_2).unsqueeze(-1)

        face_idx = torch.where(condition, face_idx_1, face_idx_2)
        tbn = self.tbn[face_idx]

        if h_threshold is None:
            h_threshold = np.inf
        h_mask = (sdf.abs() < min(self.depth_threshold, h_threshold)).squeeze(-1)
        if requires_grad_xyz:
            xyz, p_sur, sdf, normal = diff_project_layer.apply(xyz, p_sur, sdf, normal)
        return p_sur, sdf, h_mask, normal, tbn
        
    def weighted_project(self, xyz, K=8, return_psur=False, weighting='DualD', sdf_scale=1., sdf_offset=0., dir_vec_wdist=0.05, direct_above_check=False, direct_above_threshold=1e-1):
        normal, dir_vec, indices, dis = self.knn(xyz=xyz, K=K, use_dir_vec=False, dir_vec_wdist=dir_vec_wdist, direct_above_check=direct_above_check, direct_above_threshold=direct_above_threshold)
        sdfs = (dir_vec * normal.unsqueeze(1)).sum(-1)
        dist2D_sq = ((dir_vec - sdfs.unsqueeze(-1) * normal.unsqueeze(1)) ** 2).norm(dim=-1)
        if weighting == 'Gaussian':
            weights = torch.exp(dist2D_sq * self.gaussian_factor)
        elif weighting == 'Shepard':
            weights = 1 / (dist2D_sq + 1e-5)
        elif weighting == 'DualD':
            dk = dist2D_sq.max(dim=-1, keepdim=True)[0]
            d1 = dist2D_sq.min(dim=-1, keepdim=True)[0]
            weights = (dk - dist2D_sq) / (dk - d1 + 1e-5) * (dk + d1) / (dk + dist2D_sq)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-5)
        sdf = (sdfs * weights).sum(-1, keepdim=True) / max(1e-5, sdf_scale) - sdf_offset
        if return_psur:
            p_sur = xyz - sdf * normal
            return sdf, p_sur, normal
        return sdf, indices, weights, normal, dis

    def knn(self, xyz, K=8, use_dir_vec=True, dir_vec_wdist=0.05, weighting='Shepard', nn_consis_check=False, direct_above_check=False, direct_above_threshold=1e-1):
        K = min(K, self.max_K)
        dis, indices, _, _ = frnn.frnn_grid_points(xyz.unsqueeze(0), self.mesh_vertices.unsqueeze(0), None, None, K=K, r=self.radius, grid=self.grid, return_nn=False, return_sorted=True)
        indices = indices.detach().squeeze(0)
        dis = dis.sqrt()[0]
        normals = self.vertex_normals[indices]
        dir_vec_ori = xyz.unsqueeze(-2) - self.mesh_vertices[indices]
        dir_vec = dir_vec_ori / (dir_vec_ori.norm(dim=-1, keepdim=True) + 1e-5)

        if nn_consis_check:
            dir_vec_cosine = (dir_vec * dir_vec[..., :1, :]).sum(-1)
            dis = torch.where(dir_vec_cosine > 0, dis, 1e5 * torch.ones_like(dis))

        if direct_above_check:
            p2n_dis_min = 2 * torch.cross(normals, dir_vec, -1).norm(dim=-1).min(dim=-1)[0]
            direct_above_mask = (p2n_dis_min < direct_above_threshold).reshape([-1, 1])
            dis = torch.where(direct_above_mask, dis, 1e5 * torch.ones_like(dis))
            dir_vec_ori = torch.where(direct_above_mask.unsqueeze(-1), dir_vec_ori, 1e5 * torch.ones_like(dir_vec_ori))

        if use_dir_vec:
            weights_invd = 1 / (dis + 1e-7)
            weights_invd = weights_invd.squeeze(0)
            mean_dir_vec = (weights_invd.unsqueeze(-1) * dir_vec).sum(1, keepdims=True)
            normal_test = normals.mean(1, keepdims=True)
            mean_dir_vec = torch.where((mean_dir_vec * normal_test).sum(dim=-1, keepdims=True) < 0, - mean_dir_vec, mean_dir_vec)
            mean_dir_vec = mean_dir_vec / (mean_dir_vec.norm(dim=-1, keepdim=True) + 1e-5)
            normals = torch.cat([normals, mean_dir_vec], dim=1)
            dir_vec_wdist = np.clip(dir_vec_wdist, 1e-5, np.inf)
            dis = torch.cat([dis, dir_vec_wdist * torch.ones_like(dis[:, :1])], dim=1)
        
        if weighting == 'Gaussian':
            weights = torch.exp(dis * self.gaussian_factor)
        elif weighting == 'Shepard':
            weights = 1 / (dis + 1e-7)
        elif weighting == 'DualD':
            dk = dis.max(dim=-1, keepdim=True)[0]
            d1 = dis.min(dim=-1, keepdim=True)[0]
            weights = (dk - dis) / (dk - d1 + 1e-5) * (dk + d1) / (dk + dis)
        else:
            print('Unkonwn weighting method: ', weighting)
            exit(0)

        weights = weights / torch.sum(weights, dim=-1, keepdims=True)
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-5)
        normal = (normals * weights.unsqueeze(-1)).sum(-2)
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-5)

        return normal, dir_vec_ori, indices, dis

    def barycentric_mapping(self, xyz, normal, h_threshold=None, sdf_scale=1., sdf_offset=0., requires_grad_xyz=False, return_face_id=False):
        # direction test
        p_sur_1, _, depth_1, face_idx_1 = self.raytracer.trace(xyz, normal)  # inner
        p_sur_2, _, depth_2, face_idx_2 = self.raytracer.trace(xyz, -normal)  # outer
        condition = depth_1 < depth_2
        sdf = torch.where(condition, -depth_1, depth_2).unsqueeze(-1) / max(1e-5, sdf_scale) - sdf_offset
        p_sur = torch.where(condition.unsqueeze(-1), p_sur_1, p_sur_2)
        face_idx = torch.where(condition, face_idx_1, face_idx_2)

        if requires_grad_xyz:
            xyz, p_sur, sdf, normal = diff_project_layer.apply(xyz, p_sur, sdf, normal)

        if h_threshold is None:
            h_threshold = np.inf
        h_mask = (sdf.abs() < min(self.depth_threshold, h_threshold)).squeeze(-1)
        face_mask = face_idx == -1
        h_mask = torch.logical_and(h_mask, torch.logical_not(face_mask))

        face_idx[face_mask] = 0
        vertex_idx = self.faces[face_idx]
        triangles = self.mesh_vertices[vertex_idx]
        barycentric = points_to_barycentric(triangles, p_sur)
        if return_face_id:
            return vertex_idx, barycentric, sdf, h_mask, face_idx
        else:
            return vertex_idx, barycentric, sdf, h_mask
    
    def query_tbn(self, xyz, K=8, h_threshold=None, sdf_scale=1., sdf_offset=0.):
        normal, _, _, _ = self.knn(xyz=xyz, K=K, use_dir_vec=False, weighting='DualD', nn_consis_check=True)
        _, _, _, h_mask, face_idx = self.barycentric_mapping(xyz=xyz, normal=normal, h_threshold=h_threshold, sdf_scale=sdf_scale, sdf_offset=sdf_offset, return_face_id=True)
        tbn = self.tbn[face_idx]
        return tbn, h_mask

    def uvh(self, xyz, K=8, h_threshold=None, sdf_scale=1., sdf_offset=0., requires_grad_xyz=False, normal=None):
        if normal is None:
            normal, _, _, _ = self.knn(xyz=xyz, K=K, use_dir_vec=False, weighting='DualD', nn_consis_check=True)
        vertex_idx, barycentric, sdf, h_mask, face_idx = self.barycentric_mapping(xyz=xyz, normal=normal, h_threshold=h_threshold, sdf_scale=sdf_scale, sdf_offset=sdf_offset, requires_grad_xyz=requires_grad_xyz, return_face_id=True)
        uv = (self.uvs[vertex_idx] * barycentric.unsqueeze(-1)).sum(-2)
        uvh = torch.cat([uv, sdf], dim=-1)
        tbn = self.tbn[face_idx] if self.tbn is not None else None
        return uvh, h_mask, normal, tbn


class MeshFeatureField(nn.Module):
    def __init__(self, mesh_path, feature_dim=16, hash=True, h_threshold=0.1, K=8, bound=1, clustering=True, prob_model=True, pred_normal=True, use_lip_mlp_for_normal=True, pattern_rate=1/50, num_level=8, bound_output_normal=False):
        super().__init__()
        self.mesh_path = mesh_path

        self.h_threshold = h_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.clustering = clustering
        self.prob_model = prob_model
        self.use_lip_mlp_for_normal = use_lip_mlp_for_normal
        self.pattern_rate = pattern_rate

        self.hash = hash
        if hash:
            # Use hash grid to store surface features
            grid_name = "hashgrid_clustering" if clustering else "hashgrid"
            self.encoder, self.encoder_f_out_dim = get_encoder(grid_name, desired_resolution=1024, input_dim=3, num_levels=num_level, level_dim=2, base_resolution=512, log2_hashmap_size=19, align_corners=True)
            if self.prob_model:
                self.encoder_var, self.encoder_f_out_dim_var = get_encoder("hashgrid", desired_resolution=1024, input_dim=3, num_levels=num_level, level_dim=2, base_resolution=512, log2_hashmap_size=19, align_corners=True)
                self.encoder_var.reset_parameters(std=1e-5)
            self.need_update = False
        else:
            # Store surface features on mesh vertices
            self.feature_dim = feature_dim

            self.target_vnum = 128**2
            self.level_num = 1
            base_vnum = trimesh.load_mesh(self.mesh_path).vertices.shape[0]
            self.levels_vnum = np.stack([base_vnum * 2 **i for i in np.linspace(0, np.log2(self.target_vnum / base_vnum), self.level_num)]) if self.level_num > 1 else np.array([self.target_vnum])
            self.register_buffer('current_level_', torch.tensor(-1).long().to(self.device))

            fea_mesh_path = self.subdivide_mesh(min_vnum=128**2, obj_suffix=f'fea_level{self.current_level}')
            self.meshprojector_fea = MeshProjector(device=self.device, mesh_path=fea_mesh_path, store_f=True)

            self.post_activation = torch.nn.Tanh()
            # self.post_activation = torch.nn.Identity()
            self.encoder, self.encoder_f_out_dim = get_encoder('frequency', input_dim=self.feature_dim, multires=8)
            self.need_update = True
            self.update()
            self.clustering_layer = ClusteringLayer(n_clusters=16, hidden=self.feature_dim) if clustering else None
        
        # Encoder for SDF
        self.encoder_z, self.encoder_z_outdim = get_encoder('frequency', input_dim=1, multires=12)

        # Normal Net
        self.pred_normal = pred_normal
        if pred_normal:
            self.normal_net = Factorized_Normal_Net(x_dim=self.encoder_f_out_dim, z_dim=self.encoder_z_outdim, lip=self.use_lip_mlp_for_normal, direct_pred_coor=False, bound_output=bound_output_normal)
            list(self.normal_net.parameters())[0].data.uniform_(0, 1e-3)
        else:
            self.normal_net = None
        
        # Meshprojector for knn, mapping and normal calculation
        self.meshprojector = MeshProjector(device=self.device, mesh_path=self.mesh_path, store_f=True, store_uv=True)
        self.meshprojector_imported = None
        self.K = K
        self.bound = bound
        self.imported = False
        self.imported_type = None
        
        self.features_imported = None
        self.phi_embed_imported = None
        self.sample_tbn_inv_imported = None
        self.sample_tbn_ids_imported = None
        self.local_tbn_imported = None
        self.bounds = None
        self.rescale = False
        
        self.sdf_scale_factor = 1.
        self.sdf_offset = 0.
        self.uv_utilize_rate = 1.
        self.K_for_uv = 5

    def forward(self, x, no_import=False, no_noise=False, requires_grad_xyz=False, return_phi_embed=False, return_rot_angles=False):
        # Local coordinate systems for old shape and new shape
        local_tbn, sample_tbn_inv, new_tbn, normal_fine_local = None, None, None, None
        if not self.imported or no_import or self.imported_type is None:
            # No import mode
            p_sur, sdf, h_mask, normal_coarse, local_tbn = self.meshprojector.project(x, K=self.K, h_threshold=self.h_threshold, requires_grad_xyz=requires_grad_xyz)
            if self.hash:
                # Embed surface coordinates with hash grid
                x_embed = self.encoder(p_sur, bound=self.bound)
                if self.prob_model:
                    x_embed_var = self.encoder_var(p_sur, bound=self.bound)
                    noise = torch.zeros_like(x_embed_var) if no_noise else torch.randn_like(x_embed_var)
                    x_embed = x_embed + noise * torch.exp(x_embed_var)
            else:
                # Embed surface points with barycentric features weighting
                vertex_idx, barycentric, sdf, h_mask = self.meshprojector_fea.barycentric_mapping(x, normal_coarse, h_threshold=self.h_threshold, requires_grad_xyz=requires_grad_xyz)
                features = (self.features[vertex_idx] * barycentric.unsqueeze(-1)).sum(-2)
                x_embed = self.encoder(self.post_activation(features))
            z_embed = self.encoder_z(sdf)
            if self.pred_normal:
                # Factorized Normal Estimation
                normal_fine_local = self.normal_net(p_sur=p_sur, z_embed=z_embed, x_embed=x_embed)
                if return_phi_embed:
                    phi_embed = self.normal_net.phi_embedding(p_sur=p_sur)
                if return_rot_angles:
                    theta, phi = self.normal_net(p_sur=p_sur, z_embed=z_embed, x_embed=x_embed, return_rot_angles=True)
        else:
            # Import mode
            if self.imported_type == 'field':
                rescale = self.rescale
                if rescale:
                    p_sur = x[..., :2] * self.uv_utilize_rate
                    sdf = x[..., -1:] * self.bounds[0] * self.uv_utilize_rate
                else:
                    p_sur = torch.zeros_like(x[..., :2])
                    p_sur[..., 0] = x[..., 0] / self.bounds[0]
                    p_sur[..., 1] = x[..., 1] / self.bounds[1]
                    sdf = x[..., -1:]
                sdf = sdf - self.sdf_offset
                h_mask = sdf[..., 0].abs() < self.h_threshold
                # Fields on xy plane. Take O(1) to query features.
                h_mask = torch.logical_and(h_mask, (p_sur[..., :2] >= -1.).all(dim=-1))
                h_mask = torch.logical_and(h_mask, (p_sur[..., :2] <= 1.).all(dim=-1))
                x_embed = torch.nn.functional.grid_sample(self.features_imported, p_sur[None, None, :, :2], align_corners=True, padding_mode="zeros").squeeze().permute(1, 0)
                if not self.hash:
                    x_embed = self.encoder(self.post_activation(x_embed))
                z_embed = self.encoder_z(sdf)
                normal_coarse = torch.zeros_like(x)
                normal_coarse[..., -1] = 1.
                
                if self.pred_normal:
                    sample_tbn_id = torch.nn.functional.grid_sample(self.sample_tbn_ids_imported, p_sur[None, None, :, :2], mode='nearest', align_corners=True, padding_mode="zeros").squeeze().long()
                    sample_tbn_inv = self.sample_tbn_inv_imported[sample_tbn_id]
                    local_tbn = torch.nn.functional.grid_sample(self.local_tbn_imported, p_sur[None, None, :, :2], mode='nearest', align_corners=True, padding_mode="zeros").squeeze().permute(1, 0).reshape([-1, 3, 3])
                    phi_embed = torch.nn.functional.grid_sample(self.phi_embed_imported, p_sur[None, None, :, :2], align_corners=True, padding_mode="zeros").squeeze().permute(1, 0)
                    normal_fine_local = self.normal_net(phi_embed=phi_embed, z_embed=z_embed, x_embed=x_embed)
            elif self.imported_type == 'patch':
                # Patch resampled
                sdf, idx, weights, normal_coarse, dis = self.meshprojector_imported.weighted_project(x, K=8, direct_above_check=True, direct_above_threshold=1.)
                knn_features = self.features_imported.reshape([-1, self.features_imported.shape[-1]])[idx.reshape([-1])].reshape([*idx.shape, -1])
                x_embed = (weights.unsqueeze(-1) * knn_features).sum(-2)
                if not self.hash:
                    x_embed = self.encoder(self.post_activation(x_embed))
                z_embed = self.encoder_z(sdf)
                h_mask = sdf[..., 0].abs() < self.h_threshold
                h_mask = torch.logical_and(h_mask, dis.min(dim=-1)[0] < self.h_threshold)

                if self.pred_normal:
                    knn_features_phi = self.phi_embed_imported.reshape([-1, self.phi_embed_imported.shape[-1]])[idx.reshape([-1])].reshape([*idx.shape, -1])
                    phi_embed = (weights.unsqueeze(-1) * knn_features_phi).sum(-2)
                    knn_local_tbn = self.local_tbn_imported.reshape([-1, 9])[idx.reshape([-1])].reshape([*idx.shape, 3, 3])
                    local_tbn = (weights.unsqueeze(-1).unsqueeze(-1) * knn_local_tbn).sum(-3)
                    normal_fine_local = self.normal_net(phi_embed=phi_embed, z_embed=z_embed, x_embed=x_embed)
            elif self.imported_type == 'shape':
                # Shape
                uvh, h_mask, normal_coarse, new_tbn = self.meshprojector_imported.uvh(x, K=self.K_for_uv, h_threshold=self.h_threshold, sdf_scale=self.sdf_scale_factor / self.uv_utilize_rate, sdf_offset=self.sdf_offset, requires_grad_xyz=requires_grad_xyz)
                p_sur = uvh[..., :2] * self.uv_utilize_rate  # in [-1, 1]
                x_embed = torch.nn.functional.grid_sample(self.features_imported, p_sur[None, None, :, :2], align_corners=True, padding_mode="zeros").squeeze().permute(1, 0)
                if not self.hash:
                    x_embed = self.encoder(self.post_activation(x_embed))
                z_embed = self.encoder_z(uvh[..., -1:])

                if self.pred_normal:
                    sample_tbn_id = torch.nn.functional.grid_sample(self.sample_tbn_ids_imported, p_sur[None, None, :, :2], mode='nearest', align_corners=True, padding_mode="zeros").squeeze().long()
                    sample_tbn_inv = self.sample_tbn_inv_imported[sample_tbn_id]
                    local_tbn = torch.nn.functional.grid_sample(self.local_tbn_imported, p_sur[None, None, :, :2], mode='nearest', align_corners=True, padding_mode="zeros").squeeze().permute(1, 0).reshape([-1, 3, 3])
                    phi_embed = torch.nn.functional.grid_sample(self.phi_embed_imported, p_sur[None, None, :, :2], align_corners=True, padding_mode="zeros").squeeze().permute(1, 0)
                    normal_fine_local = self.normal_net(phi_embed=phi_embed, z_embed=z_embed, x_embed=x_embed)
            elif self.imported_type == 'unhash':
                # Resample to mesh
                normal_coarse, _, _, _ = self.meshprojector.knn(x)
                local_tbn = self.meshprojector.query_tbn(x)
                vertex_idx, barycentric, sdf, h_mask = self.meshprojector_imported.barycentric_mapping(x, normal=normal_coarse, h_threshold=self.h_threshold, requires_grad_xyz=requires_grad_xyz, sdf_scale=self.sdf_scale_factor, sdf_offset=self.sdf_offset)
                x_embed = (self.features_imported[vertex_idx] * barycentric.unsqueeze(-1)).sum(-2)
                z_embed = self.encoder_z(sdf)

                if self.pred_normal:
                    phi_embed = (self.phi_embed_imported[vertex_idx] * barycentric.unsqueeze(-1)).sum(-2)
                    normal_fine_local = self.normal_net(phi_embed=phi_embed, z_embed=z_embed, x_embed=x_embed)
        embed = torch.cat([x_embed, z_embed], dim=-1)
        normal_coarse = normal_coarse / (normal_coarse.norm(dim=-1, keepdim=True) + 1e-5)

        normal_fine = normal_fine_local
        if self.pred_normal:
            if local_tbn is not None:
                normal_fine = torch.einsum('nba,nb->na', local_tbn, normal_fine)
            if sample_tbn_inv is not None:
                normal_fine = torch.einsum('nba,nb->na', sample_tbn_inv, normal_fine)
            if new_tbn is not None:
                normal_fine = torch.einsum('nba,nb->na', new_tbn, normal_fine)
            normal_fine = normal_fine / (normal_fine.norm(dim=-1, keepdim=True) + 1e-5)
        
        if self.pred_normal and return_phi_embed:
            return embed, normal_coarse, normal_fine, h_mask, phi_embed
        elif self.pred_normal and return_rot_angles:
            return embed, normal_coarse, normal_fine, h_mask, theta, phi
        else:
            return embed, normal_coarse, normal_fine, h_mask
    
    @property
    def current_level(self):
        return self.current_level_.item()

    @current_level.setter
    def current_level(self, current_level: int):
        self.register_buffer('current_level_', torch.tensor(current_level).long().to(self.device))
    
    def clustering_loss(self):
        if self.clustering:
            if self.hash:
                cl_loss = self.encoder.clustering_loss()
            else:
                cl_loss = self.clustering_layer.clustering_loss(self.features)
            return cl_loss
        else:
            return 0.
    
    def kl_loss(self, normal=False):
        if self.prob_model:
            f_mu = self.encoder.embeddings
            f_var = self.encoder_var.embeddings
            if normal:
                kl_loss = 0.5 * (torch.exp(f_var) + f_mu**2 - 1. - f_var).sum()
            else:
                kl_loss = 0.5 * (torch.exp(f_var) - 1. - f_var).sum()
            return kl_loss
        else:
            return 0.
    
    def regular_loss(self):
        cl_loss = self.clustering_loss()
        kl_loss = self.kl_loss()
        # return 1e-5 * cl_loss + 1e-3 * kl_loss # For common cases
        return 1e-8 * cl_loss  # For glossy surface
    
    def uv(self, x):
        meshprojector = self.meshprojector_imported if self.imported else self.meshprojector
        if meshprojector is None:
            return None, None
        if meshprojector.uvs is not None:
            sdf_scale_factor = self.sdf_scale_factor if self.imported_type == 'shape' else 1.
            sdf_offset = self.sdf_offset if self.imported_type == 'shape' else 0.
            uvh, h_mask, _, _ = meshprojector.uvh(x, K=self.K_for_uv, h_threshold=self.h_threshold, sdf_scale=sdf_scale_factor, sdf_offset=sdf_offset)
        else:
            uvh = torch.zeros_like(x)
            h_mask = torch.zeros_like(x[..., 0], dtype=torch.bool)
        return uvh, h_mask
    
    def tbn(self, x):
        tbn = torch.zeros([*x.shape[:-1], 3, 3], dtype=x.dtype, device=x.device)
        if self.imported_type == 'field' and self.imported:
            for i in range(3):
                tbn[..., i, i] = 1.
        meshprojector = self.meshprojector_imported if self.imported else self.meshprojector
        if meshprojector is not None and meshprojector.tbn is not None:
            sdf_scale_factor = self.sdf_scale_factor if self.imported_type == 'shape' else 1.
            sdf_offset = self.sdf_offset if self.imported_type == 'shape' else 0.
            tbn, _ = meshprojector.query_tbn(x, K=self.K_for_uv, h_threshold=self.h_threshold, sdf_scale=sdf_scale_factor, sdf_offset=sdf_offset)
        return tbn

    def subdivide_mesh(self, min_vnum=5e5, obj_suffix='fine', replace=False):
        obj_path = '.'.join(self.mesh_path.split('.')[:-1]) + '_' + obj_suffix + '.obj'
        if not os.path.exists(obj_path) or replace:
            mesh = trimesh.load_mesh(self.mesh_path)
            while mesh.vertices.shape[0] < min_vnum:
                v, f = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
                mesh = trimesh.Trimesh(vertices=v, faces=f)
            mesh.export(obj_path)
        return obj_path
    
    def update(self, target_level=None):
        if not self.need_update:
            return
        if self.current_level >= self.level_num - 1:
            return False
        if target_level is not None and self.current_level >= target_level:
            return False
        if target_level is None:
            target_level = self.current_level + 1
        if self.current_level < self.level_num - 1:
            print(f'Update mesh feature field from level {self.current_level} to {self.current_level+1}')
            self.current_level = int(max(self.current_level + 1, min(target_level, self.level_num-1)))
            fea_mesh_path = self.subdivide_mesh(min_vnum=self.levels_vnum[self.current_level], obj_suffix=f'fea_level_{self.current_level}', replace=True)
            fea_mesh = trimesh.load_mesh(fea_mesh_path)
            features = torch.empty((fea_mesh.vertices.shape[0], self.feature_dim), dtype=torch.float32, device=self.device)
            print('Current features shape: ', features.shape)
            if self.current_level == 0:
                torch.nn.init.uniform_(features, -1., 1.)
            else:
                vertices = torch.from_numpy(fea_mesh.vertices).float().to(self.device)
                start = 0
                while start < vertices.shape[0]:
                    end = min(start + 4096, vertices.shape[0])
                    _, _, _, normal, _ = self.meshprojector.project(vertices[start: end], K=self.K, h_threshold=self.h_threshold)
                    vertex_idx, barycentric, _, _ = self.meshprojector_fea.barycentric_mapping(vertices[start: end], normal, h_threshold=self.h_threshold)
                    features_ = (self.features[vertex_idx] * barycentric.unsqueeze(-1)).sum(-2)
                    features[start: end] = features_
                    start = end
            self.features = nn.Parameter(features)
            self.meshprojector_fea = MeshProjector(device=self.device, mesh_path=fea_mesh_path, store_f=True)
            print('Update done!')
    
    def unhash(self):
        print('Unhashing ...')
        unhash_obj_path = self.subdivide_mesh(min_vnum=5e5, obj_suffix='unhash')
        self.meshprojector_imported = MeshProjector(device=self.device, mesh_path=unhash_obj_path, store_f=True)
        mesh_fine = trimesh.load_mesh(unhash_obj_path)
        vertices_fine = torch.from_numpy(mesh_fine.vertices).float().to(self.device)
        features = []
        start = 0
        while start < vertices_fine.shape[0]:
            end = min(start + 4096, vertices_fine.shape[0])
            features_ = self.encoder(vertices_fine[start: end], bound=self.bound)
            features.append(features_)
            start = end
        features = torch.cat(features, dim=0)
        self.features_imported = nn.Parameter(features)
        self.imported = True
        self.imported_type = 'unhash'
        print('Unhashed!')
        torch.cuda.empty_cache()
    
    def import_unhash_vertices(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        # self.meshprojector_imported = MeshProjector(device=self.device, mesh=data['mesh'][()], store_f=True, compute_normals=True, ini_raytracer=True)
        from .map_bvh import BvhMeshProjector
        self.meshprojector_imported = BvhMeshProjector(device=self.device, mesh=data['mesh'][()], store_f=True, compute_normals=True, ini_raytracer=True)
        features = data['features']
        self.features_imported = nn.Parameter(torch.from_numpy(features).float().to(self.device))
        self.imported = True
        self.imported_type = 'unhash'
        self.sdf_scale_factor = float(data['sdf_factor'])
        print('Unhashed Synthesized Results! sdf factor: ', self.sdf_scale_factor)
        torch.cuda.empty_cache()
    
    def import_unhash(self, data_path, res=2048):
        data = np.load(data_path, allow_pickle=True)
        features = torch.from_numpy(data['features'])
        features_imported = torch.zeros([res, res, features.shape[-1]], dtype=torch.float32).reshape([-1, features.shape[-1]]).to(self.device)
        # Simple Mesh Projector
        from .map_bvh import BvhMeshProjector
        self.meshprojector_imported = BvhMeshProjector(device=self.device, mesh=data['simple_mesh'][()], compute_normals=True, ini_raytracer=True, store_f=True, store_uv=True)
        # Plane Atlas Grid
        from copy import deepcopy
        mesh_plane = deepcopy(self.meshprojector_imported.mesh)
        mesh_plane.vertices = np.zeros_like(mesh_plane.vertices)
        mesh_plane.vertices[:, :2] = self.meshprojector_imported.uvs.cpu().numpy()
        meshprojector_plane = BvhMeshProjector(device=self.device, mesh=mesh_plane, store_f=True, compute_normals=True, ini_raytracer=True)
        # Features-Vertices Mesh
        meshprojector_complex = BvhMeshProjector(device=self.device, mesh=data['mesh'][()], store_f=True, compute_normals=True, ini_raytracer=True)
        # Traverse UV map
        print('Doing features moving...')
        us, vs = torch.meshgrid(torch.linspace(-1, 1, res), torch.linspace(-1, 1, res), indexing='xy')
        uvs = torch.stack([us, vs, torch.zeros_like(us)], dim=-1).reshape([-1, 3]).to(self.device)
        batch, start = 2048, 0
        while start < uvs.shape[0]:
            end = min(start + batch, uvs.shape[0])
            sdf, fids, barycentric = meshprojector_plane.bvh.signed_distance(uvs[start: end], return_uvw=True, mode='raystab')
            verts_3d = meshprojector_plane.barycentric_weighting(vert_values=self.meshprojector_imported.mesh_vertices, fids=fids, barycentric=barycentric)
            vids, barycentric_complex, sdf_3d, _ = meshprojector_complex.barycentric_mapping(verts_3d)
            features_imported[start: end] = (features[vids].to(self.device) * barycentric_complex.unsqueeze(-1)).sum(-2) * (sdf[..., None] < 0.1).float()
            start = end
        print('Features moving done! Feature map resolution: ', res)
        self.features_imported = nn.Parameter(features_imported.float().to(self.device).reshape([1, res, res, -1]).permute(0, 3, 1, 2))
        self.imported = True
        self.imported_type = 'shape'
        self.sdf_scale_factor = float(data['sdf_factor'])
        self.bounds = [.5 * data['original_grid_gap'] * res, .5 * data['original_grid_gap'] * res]
        self.rescale = True
        print('Unhashed Synthesized Results! sdf factor: ', self.sdf_scale_factor)
        torch.cuda.empty_cache()
    
    def import_field(self, features, bounds, sample_tbn, sample_tbn_ids, local_tbn, phi_embed):
        self.vnum = features.shape[0]
        self.bounds = bounds
        features = np.moveaxis(features[None], -1, 1)  # From H,W,C to 1,C,H,W
        phi_embed = np.moveaxis(phi_embed[None], -1, 1)
        local_tbn = np.moveaxis(local_tbn[None], -1, 1)
        sample_tbn_ids = np.moveaxis(sample_tbn_ids[None, ..., None], -1, 1)  # From H,W to 1,C,H,W
        self.features_imported = torch.from_numpy(features).permute(0, 1, 3, 2).to(self.device).float()  # Permute to let uv correspond to xy
        self.phi_embed_imported = torch.from_numpy(phi_embed).permute(0, 1, 3, 2).to(self.device).float()
        self.sample_tbn_inv_imported = torch.from_numpy(sample_tbn).to(self.device).float().reshape([-1, 3, 3]).inverse()
        self.sample_tbn_ids_imported = torch.from_numpy(sample_tbn_ids).permute(0, 1, 3, 2).to(self.device).float()
        self.local_tbn_imported = torch.from_numpy(local_tbn).permute(0, 1, 3, 2).to(self.device).float()
        self.imported = True
        self.imported_type = 'field'
        self.rescale = not self.rescale
        torch.cuda.empty_cache()
    
    def import_patch(self, features, mesh, local_tbn, phi_embed):
        self.vnum = features.shape[0]
        self.meshprojector_imported = MeshProjector(device=self.device, mesh=mesh, mean_edge_length=self.meshprojector.mean_edge_length, compute_normals=False, ini_raytracer=False, store_uv=False)
        self.features_imported = torch.from_numpy(features).to(self.device).float()
        self.local_tbn_imported = torch.from_numpy(local_tbn).to(self.device).float()
        self.phi_embed_imported = torch.from_numpy(phi_embed).to(self.device).float() if phi_embed is not None else None
        self.imported = True
        self.imported_type = 'patch'
        torch.cuda.empty_cache()

    def import_shape(self, mesh):
        if not (self.imported_type == 'field' or self.imported_type == 'shape'):
            print('Need to load field firstly !!!')
            return
        self.meshprojector_imported = MeshProjector(device=self.device, mesh=mesh, compute_normals=True, ini_raytracer=True, store_f=True, store_uv=True)
        if self.meshprojector_imported.recommended_sdf_factor is not None:
            self.sdf_scale_factor = self.meshprojector_imported.recommended_sdf_factor / self.bounds[0]
            print('Recommend sdf factor: ', self.sdf_scale_factor)
        self.imported = True
        self.imported_type = 'shape'
        torch.cuda.empty_cache()

    def sample_patches(self, patch_size=128, max_patch_num=2000, scan_pcl_path=None, sample_on_template=False, sample_on_picked_faces=True, picked_faces_path=None, use_trimesh_raycast=False, sample_poisson_disk=True, record_rays=False, work_space='./test_data/', cast_on_mfs=True):

        # Build kd tree of scan point cloud for nn query
        scan_pcl = trimesh.load_mesh(scan_pcl_path).vertices if scan_pcl_path is not None else None
        scan_tree = KDTree(scan_pcl, leaf_size=2)

        # Load template mesh and determine grid gap
        mesh = trimesh.load_mesh(self.mesh_path)
        # Sample on picked faces or the whole mesh
        if sample_on_picked_faces and picked_faces_path is not None and os.path.exists(picked_faces_path):
            print('Sampling on ', picked_faces_path)
            mesh_for_sample = trimesh.load(picked_faces_path)
        else:
            print('Sampling on the whole mesh')
            mesh_for_sample = mesh
        edges = mesh_for_sample.vertices[mesh_for_sample.edges_unique]
        edges = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=-1)
        mean_edge_length = edges.mean()
        grid_gap = mean_edge_length * self.pattern_rate
        patch_edge_length = patch_size * grid_gap

        # Sampled patch align with the first component
        pca = PCA(n_components=3)
        if os.path.exists(work_space + '/meshes/direction.obj'):
            print('Use direction.obj for PCA')
            vertices_dir = trimesh.load_mesh(work_space + '/meshes/direction.obj').vertices
        else:
            print(f'Use {self.mesh_path} for PCA')
            vertices_dir = mesh.vertices
        pca.fit(vertices_dir)
        first_component = pca.components_[0]  # np.array([1., 0., 0.])
        print('First component: ', first_component)

        # Initialize patch coordinates
        calibration = np.linspace(-patch_size*grid_gap/2, patch_size*grid_gap/2, patch_size)
        x, y = np.meshgrid(calibration, calibration, indexing='ij')
        patch_coor = np.stack([x, y], axis=-1).reshape([-1, 2])
        patch_coor = np.concatenate([patch_coor, np.zeros_like(patch_coor)], axis=-1)
        patch_coor[..., -1] = 1
        if not os.path.exists(work_space+'/meshes/'):
            os.makedirs(work_space+'/meshes/')
        write_ply(patch_coor, work_space + '/meshes/sample_patch.ply')
        if not use_trimesh_raycast:
            patch_coor = torch.from_numpy(patch_coor).float().to(self.device)
            # cast_on_mfs determines whether cast on mesh_for_sample or not
            raytracer = RayTracer(mesh_for_sample.vertices, mesh_for_sample.faces) if cast_on_mfs else RayTracer(mesh.vertices, mesh.faces)
        
        # Estimate template normals for projection
        mesh_for_sample.as_open3d.compute_vertex_normals()
        if sample_on_template:
            v_normals = np.asarray(mesh_for_sample.as_open3d.vertex_normals)
            vertices = np.asarray(mesh_for_sample.vertices)
        elif sample_poisson_disk:
            vertices = np.asarray(mesh_for_sample.as_open3d.sample_points_poisson_disk(max_patch_num).points)
            _, _, face_idx = trimesh.proximity.closest_point(mesh_for_sample, vertices)
            v_normals = mesh_for_sample.face_normals[face_idx]
        else:
            vertices, face_idx = trimesh.sample.sample_surface_even(mesh_for_sample, 20000, radius=patch_edge_length/16)
            vertices = vertices[:min(vertices.shape[0], max_patch_num)]
            face_idx = face_idx[:min(face_idx.shape[0], max_patch_num)]
            v_normals = mesh_for_sample.face_normals[face_idx]

        # Sample patches along the surface of template mesh
        print('Getting patches from curved surface ...')
        patches = np.zeros([max_patch_num, patch_size, patch_size, self.encoder_f_out_dim])
        patch_coors = np.zeros([max_patch_num, patch_size, patch_size, 3])
        patch_norms = np.zeros([max_patch_num, 3])
        patch_sample_tbn = np.zeros([max_patch_num, 9])
        picked_vertices = np.zeros([max_patch_num,  3])
        patch_phi_embed = np.zeros([max_patch_num, patch_size, patch_size, self.normal_net.encoder_out_dim]) if self.normal_net is not None else None
        patch_local_tbn = np.zeros([max_patch_num, patch_size, patch_size, 9])
        patch_rays = np.zeros([max_patch_num, patch_size, patch_size, 6])
        count = 0
        for i in tqdm(range(vertices.shape[0])):
            # Discard parts below y=0 when no scan_pcl
            if scan_pcl is None and vertices[i, 1] < 0:
                continue

            # Determine the transform matrix by sample vertex
            z_axis = v_normals[i]
            y_axis = np.cross(z_axis, first_component)
            if y_axis.sum() == 0:
                y_axis = np.cross(z_axis, np.array([1., 1., 1.01]) * first_component)
            y_axis = y_axis / np.linalg.norm(y_axis)
            x_axis = np.cross(y_axis, z_axis)
            T = np.eye(4)
            T[:3, :3] = np.stack([x_axis, y_axis, z_axis], -1)
            T[:3, 3] = vertices[i]

            if use_trimesh_raycast:
                # Initialize intersections and mask of intersection
                mask = np.zeros((patch_coor.shape[0]), dtype=bool)
                intersections = np.zeros((patch_coor.shape[0], 3), dtype=np.float32)
                # Transform the patch coordinates to ray origins
                ray_origins = np.einsum('ab,nb->na', T, patch_coor)[..., :3]
                # Discard patches far from the template surface
                nn_dist, _ = scan_tree.query(ray_origins, k=1)
                if nn_dist.max() > 3 * self.h_threshold:
                    continue
                # Move ray origins away from template along the z_axis
                ray_origins +=  1e3 * z_axis
                # Ray casting
                ray_directions = np.broadcast_to(-z_axis[None], ray_origins.shape)
                # cast_on_mfs determines whether cast on mesh_for_sample or not
                mesh_for_cast = mesh_for_sample if cast_on_mfs else mesh
                locations, index_ray, _ = mesh_for_cast.ray.intersects_location(
                    ray_origins=ray_origins,
                    ray_directions=ray_directions)
                for ray_idx in np.unique(index_ray):
                    idx = np.where(index_ray == ray_idx)[0]
                    location = locations[idx]
                    zs = ((ray_origins[ray_idx] - location)**2).sum(axis=-1)
                    zmin_idx = np.argmin(zs)
                    location = location[zmin_idx]
                    intersections[ray_idx] = location
                    mask[ray_idx] = True
                intersections = torch.from_numpy(intersections).to(self.device)
                # Discard those patches with un-intersected pixels
                if not mask.all():
                    continue
            else:
                T = torch.from_numpy(T).float().to(self.device)
                ray_origins = torch.einsum('ab,nb->na', T, patch_coor)[..., :3]
                if ray_origins.isnan().any():
                    continue
                # Discard patches far from the template surface
                nn_dist, _ = scan_tree.query(ray_origins.cpu().numpy(), k=1)
                if nn_dist.max() > min(1e-1, 3 * self.h_threshold):
                    continue
                ray_origins +=  .1 * torch.from_numpy(z_axis).float().to(self.device)
                ray_directions = torch.from_numpy(np.copy(np.broadcast_to(-z_axis[None], ray_origins.shape))).float().to(self.device)
                intersections, _, depth, _ = raytracer.trace(ray_origins, ray_directions)
                if depth.max().item() > 9.5:
                    continue
            
            # Gater patches
            p_sur, _, _, normal, local_tbn = self.meshprojector.project(intersections, K=self.K, h_threshold=self.h_threshold, requires_grad_xyz=False)
            if self.hash:
                # Embed surface coordinates with hash grid
                patch = self.encoder(p_sur, bound=self.bound)
            else:
                # Embed surface points with barycentric features weighting
                vertex_idx, barycentric, _, _ = self.meshprojector_fea.barycentric_mapping(intersections, normal, h_threshold=self.h_threshold, requires_grad_xyz=False)
                patch = (self.features[vertex_idx] * barycentric.unsqueeze(-1)).sum(-2)
            patches[count] = patch.detach().cpu().numpy().reshape([patch_size, patch_size, -1])
            patch_local_tbn[count] = local_tbn.cpu().numpy().reshape([patch_size, patch_size, 9])
            patch_coors[count] = intersections.cpu().numpy().reshape([patch_size, patch_size, 3])
            patch_norms[count] = z_axis
            if patch_phi_embed is not None:
                phi_embed = self.normal_net.phi_embedding(p_sur)
                patch_phi_embed[count] = phi_embed.detach().cpu().numpy().reshape([patch_size, patch_size, -1])
            if not type(T) == np.ndarray:
                T = T.cpu().numpy()
            patch_sample_tbn[count] = T[:3, :3].T.reshape([9])
            picked_vertices[count] = vertices[i]
            if record_rays:
                if not type(ray_origins) == np.ndarray:
                    ray_origins = ray_origins.cpu().numpy()
                    ray_directions = ray_directions.cpu().numpy()
                rays = np.concatenate([ray_origins, ray_directions], axis=-1)
                patch_rays[count] = rays.reshape([patch_size, patch_size, 6])
            count += 1
            if max_patch_num is not None and count == max_patch_num:
                break

        # Stack and return
        patches = patches[:count]
        patch_coors = patch_coors[:count]
        patch_norms = patch_norms[:count]
        patch_sample_tbn = patch_sample_tbn[:count]
        patch_local_tbn = patch_local_tbn[:count]
        picked_vertices = picked_vertices[:count]
        if patch_phi_embed is not None:
            patch_phi_embed = patch_phi_embed[:count]
        if record_rays:
            patch_rays = patch_rays[:count]
        print('Get patches: ', patches.shape, ' Grid Gap: ', grid_gap)
        return patches, grid_gap, patch_coors, patch_norms, patch_sample_tbn, patch_local_tbn, picked_vertices, patch_phi_embed, patch_rays

    def sample_features(self, fea_num=5000):
        # Load template mesh and determine grid gap
        mesh = trimesh.load_mesh(self.mesh_path)
        print('Sampling on the whole mesh')
        mesh_for_sample = mesh
        vertices = np.asarray(mesh_for_sample.as_open3d.sample_points_poisson_disk(fea_num).points)
        vertices = torch.from_numpy(vertices).float().to(self.device)
        features = torch.zeros([vertices.shape[0], self.encoder.output_dim]).float().to(self.device)
        batch_size = 1024
        start = 0
        while start < vertices.shape[0]:
            end = min(start+batch_size, vertices.shape[0])
            features[start: end] = self.encoder(vertices[start: end], bound=self.bound)
            start = end
        return features

    def visualize_features(self, sv_path='./'):
        if not os.path.exists(sv_path):
            os.makedirs(sv_path)
        offsets = torch.cat([self.encoder.offsets, self.encoder.embeddings.shape[0] * torch.ones_like(self.encoder.offsets[:1])])
        for i in range(self.encoder.num_levels):
            embeddings = self.encoder.embeddings[offsets[i]: offsets[i+1]].detach().cpu().numpy()
            plt.clf()
            plt.scatter(embeddings[:, 0], embeddings[:, 1])
            cluster_centers = self.encoder.cluster_layers[i].cluster_centers.detach().cpu().numpy()
            plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c=np.arange(cluster_centers.shape[0]), cmap='viridis')
            plt.savefig(sv_path + f'/level_{i}_features.png')
