import os
import cubvh
import torch
import xatlas
import trimesh
import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from pytorch3d.structures import Meshes, Pointclouds
from tools.shape_tools import write_ply


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class BvhMeshProjector:
    def __init__(self, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), mesh_path=None, mesh=None, mean_edge_length=None, compute_normals=True, store_f=False, store_uv=False, **kwargs):
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

        # Store vertices and create frnn
        self.mesh_vertices = torch.FloatTensor(self.mesh.vertices).to(device)
        print(f'Mesh Projector with {self.mesh_vertices.shape[0]} vertices')
        self.vertex_normals = torch.FloatTensor(np.asarray(self.mesh.as_open3d.vertex_normals)).to(device)
        
        # Initialize raytracer
        self.bvh = cubvh.cuBVH(self.mesh.vertices, self.mesh.faces)
        self.depth_threshold = 9.5

        # Store faces
        if store_f:
            self.faces = torch.from_numpy(self.mesh.faces).to(device).long()
        else:
            self.faces = None

    def barycentric_weighting(self, vert_values, fids, barycentric, **kwargs):
        face_values = vert_values[self.faces[fids]]
        weighted_values = (face_values * barycentric[..., None]).sum(dim=1)
        return weighted_values

    def project(self, xyz, h_threshold=None, requires_grad_xyz=False, **kwargs):
        sdf, fid, barycentric = self.bvh.signed_distance(xyz, return_uvw=True, mode='raystab')
        barycentric = barycentric.abs()
        p_sur = self.barycentric_weighting(vert_values=self.mesh_vertices, fids=fid, barycentric=barycentric)
        h_mask = (sdf.abs() < h_threshold).squeeze(-1)
        normal = self.barycentric_weighting(vert_values=self.vertex_normals, fids=fid, barycentric=barycentric)
        tbn = self.tbn[fid]
        return p_sur, sdf, h_mask, normal, tbn
        
    def query_tbn(self, xyz, h_threshold=None, sdf_scale=1., sdf_offset=0.):
        sdf, fid, _ = self.bvh.signed_distance(xyz, return_uvw=True, mode='raystab')
        sdf = sdf / sdf_scale - sdf_offset
        tbn = self.tbn[fid]
        h_mask = (sdf.abs() < h_threshold).squeeze(-1)
        return tbn, h_mask

    def uvh(self, xyz, h_threshold=None, sdf_scale=1., sdf_offset=0., requires_grad_xyz=False, **kwargs):
        sdf, fid, barycentric = self.bvh.signed_distance(xyz, return_uvw=True, mode='raystab')
        sdf = sdf / sdf_scale - sdf_offset
        uv = self.barycentric_weighting(self.uvs, fids=fid, barycentric=barycentric)
        uvh = torch.cat([uv, sdf[..., None]], dim=-1)
        tbn = self.tbn[fid] if self.tbn is not None else None
        h_mask = (sdf.abs() < h_threshold).squeeze(-1)
        normal = self.barycentric_weighting(vert_values=self.vertex_normals, fids=fid, barycentric=barycentric)
        return uvh, h_mask, normal, tbn

    def barycentric_mapping(self, xyz, h_threshold=None, sdf_scale=1., sdf_offset=0., requires_grad_xyz=False, return_face_id=False, **kwargs):
        sdf, fid, barycentric = self.bvh.signed_distance(xyz, return_uvw=True, mode='raystab')
        sdf = sdf / sdf_scale - sdf_offset
        sdf = sdf.unsqueeze(-1)
        h_mask = (sdf.abs() < h_threshold).squeeze(-1) if h_threshold is not None else None
        vertex_idx = self.faces[fid]        
        if return_face_id:
            return vertex_idx, barycentric, sdf, h_mask, fid
        else:
            return vertex_idx, barycentric, sdf, h_mask

