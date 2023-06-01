import torch
import trimesh
import numpy as np
import open3d as o3d
import torch.nn as nn
import tinycudann as tcnn
from tools.activation import trunc_exp
from tools.encoding import get_encoder
from .renderer import NeRFRenderer
from tools.map import MeshFeatureField
from .sg_light_model import SG_EnvmapMaterialNet
from .sh_light_model import SH_EnvmapMaterialNet
from .envmap_light_model import Envmap_EnvmapMaterialNet
from scipy.spatial.transform import Rotation as R



class LowpassFilter:
    def __init__(self, encoder_f_out_dim, encoder_z_out_dim, band_len_f=8, band_len_z=6):
        self.band_len_f = min(encoder_f_out_dim, band_len_f)
        self.band_len_z = min(encoder_z_out_dim, band_len_z)
        self.encoder_f_out_dim = encoder_f_out_dim
        self.encoder_z_out_dim = encoder_z_out_dim
        self.out_dim = self.band_len_f + self.band_len_z

    def __call__(self, x_embed):
        assert x_embed.shape[-1] == self.encoder_f_out_dim + self.encoder_z_out_dim, 'The dim -1 of x_embed doest not match f, z embedding'
        low_freq_fea = torch.cat([x_embed[..., :self.band_len_f], x_embed[..., self.encoder_f_out_dim:self.encoder_f_out_dim+self.band_len_z]], dim=-1)
        return low_freq_fea


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                surface_mesh_path,
                h_threshold=.1,
                feature_dim=4,
                num_layers=2,
                hidden_dim=32,
                geo_feat_dim=15,
                num_layers_color=3,
                hidden_dim_color=64,
                bound=1,
                dir_degree=4,
                hash=True,
                cal_dist_loss=True,
                clustering=True,
                prob_model=True,
                regularization=True,
                torch_sigma_layer=False,
                light_model=None,
                num_level=8,
                # For SG Light Model
                num_lgt_sgs=8,
                numBrdfSGs=1,
                white_specular=True,
                # For SH Light Model
                sh_order=3,
                use_specular=True,
                coarse_as_primary=False,
                # For Both
                white_light=True,
                # Finer Normal
                lip_mlp=True,
                smooth_grad_weight=1e-1,
                # Texture Pattern Size Rate,
                pattern_rate=1/50,
                # Ablation
                no_visibility=False,
                # Bound Theta and Phi
                bound_output_normal=False,
                **kwargs
                ):
        super().__init__(bound, cal_dist_loss=cal_dist_loss, regularization=regularization, **kwargs)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dir_degree = dir_degree
        self.feature_dim = feature_dim
        self.surface_mesh_path = surface_mesh_path
        self.h_threshold = h_threshold
        self.clustering = clustering
        self.prob_model = prob_model
        self.hash = hash
        self.torch_sigma_layer = torch_sigma_layer
        self.use_lip_mlp_for_normal = lip_mlp
        self.shade_visibility = True
        self.use_coarse_normal = False
        self.use_grad_normal = False
        self.fc_weight = 1.
        self.smooth_grad_weight = smooth_grad_weight
        self.coarse_as_primary = coarse_as_primary
        self.no_visibility = no_visibility
        self.feature_visualize_func = None
        self.feaphi_visualize_func = None
        self.bound_output_normal = bound_output_normal

        # Define the light model for relighting
        self.light_models = ['SG', 'SH', 'Envmap', 'None']
        if light_model is not None:
            if light_model not in self.light_models:
                print('Unknown light model: ', light_model)
                exit(0)
            else:
                self.light_model = light_model
                self.field_name += ('_' + light_model)
        else:
            self.light_model = None

        # Spherical Gaussian Lighting Model
        self.geo_feat_dim = geo_feat_dim
        if self.light_model == 'SG':
            self.light_net = SG_EnvmapMaterialNet(input_dim=self.geo_feat_dim, num_lgt_sgs=num_lgt_sgs, num_base_materials=numBrdfSGs, white_light=white_light, white_specular=white_specular)
            self.light_visual_modes = ['Full', 'Specular', 'Diffuse', 'Albedo']
            self.render_light_model = True
        elif self.light_model == 'SH':
            self.light_net = SH_EnvmapMaterialNet(input_dim=self.geo_feat_dim, sh_order=sh_order, white_light=white_light, use_specular=use_specular)
            self.light_visual_modes = ['Full', 'Specular', 'Diffuse', 'Albedo']
            self.render_light_model = True
        elif self.light_model == 'Envmap':
            self.light_net = Envmap_EnvmapMaterialNet(input_dim=self.geo_feat_dim, env_res=16, white_light=white_light, use_specular=use_specular)
            self.light_visual_modes = ['Full', 'Specular', 'Diffuse', 'Albedo']
            self.render_light_model = True
        else:
            self.light_visual_modes = ['RGB']
            self.render_light_model = False
        self.light_visual_mode_id = 0
        self.light_visual_mode = self.light_visual_modes[self.light_visual_mode_id]

        # Mesh Feature Field
        self.meshfea_field = MeshFeatureField(hash=hash, mesh_path=surface_mesh_path, h_threshold=h_threshold, K=8, bound=bound, clustering=clustering, prob_model=prob_model, pred_normal=self.render_light_model, use_lip_mlp_for_normal=self.use_lip_mlp_for_normal, pattern_rate=pattern_rate, num_level=num_level, bound_output_normal=bound_output_normal)

        # Field Name
        self.field_name = 'curved_grid'
        if self.hash:
            self.field_name += '_hash'
        if self.clustering:
            self.field_name += '_clus'
        if self.prob_model:
            self.field_name += '_prob'
        if self.torch_sigma_layer:
            self.field_name += '_ts'
        if self.optimize_camera:
            self.field_name += '_optcam'
        if self.use_lip_mlp_for_normal:
            self.field_name += '_lip'
        self.field_name += ('_' + self.light_model)
        if self.no_visibility:
            self.field_name += '_novis'
        if self.bound_output_normal:
            self.field_name += '_bd'

        self.attribute_name = 'features'
        self.patch_id = 200

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder_f_outdim = self.meshfea_field.encoder_f_out_dim
        self.encoder_z_outdim = self.meshfea_field.encoder_z_outdim

        if self.torch_sigma_layer:
            self.sigma_net = FClayers(in_dim=self.encoder_z_outdim+self.encoder_f_outdim, out_dim=1 + self.geo_feat_dim, n_neurons=hidden_dim, num_layers=num_layers - 1)
        else:
            self.sigma_net = tcnn.Network(
                n_input_dims=self.encoder_z_outdim+self.encoder_f_outdim,
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim,
                    "n_hidden_layers": num_layers - 1,
                },
            )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color

        # Static Lighting Model
        if not self.render_light_model:
            # Direction Embedding
            if self.dir_degree > 0:
                self.encoder_dir = tcnn.Encoding(
                    n_input_dims=3,
                    encoding_config={
                        "otype": "SphericalHarmonics",
                        "degree": self.dir_degree,
                    },
                )
                self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim
            else:
                self.in_dim_color = self.geo_feat_dim
            # Color Network
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_color,
                    "n_hidden_layers": num_layers_color - 1,
                },
            )
        else:
            self.color_net = None

        # Visual modes
        self.visual_modes = ['RGB', 'UV', 'Grad', 'Nc', 'Tan', 'Btan', 'normal', 'Fea']
        self.visual_mode_id = 0
        self.visual_mode = self.visual_modes[self.visual_mode_id]

        # Normal Network
        if self.render_light_model:
            self.visual_modes.append('Nf')

    def regular_loss(self, step):
        loss = self.meshfea_field.regular_loss()
        if self.optimize_camera:
            loss_cam = self.camera_regularization()
            weight = 1e2 if step > 2000 else 1e4
            loss = loss + weight * loss_cam
        if self.use_lip_mlp_for_normal and self.render_light_model:
            loss_lip = self.meshfea_field.normal_net.regularization()
            loss = loss + 1e-4 * loss_lip
        return loss
        
    def forward(self, x, d, euler=None, is_gui_mode=False, composite_normal_error=False, frame_index=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # Sigma
        # case: visual gradient mode; use SH for lighting while training;
        if (self.visual_mode == 'Grad' and is_gui_mode) or (self.render_light_model and self.training) or self.use_grad_normal:  #(Use Surface Normal Instead)
            # Normal mode
            with torch.enable_grad():
                x.requires_grad_(True)
                x_embed, normal_coarse, normal_fine, h_mask = self.meshfea_field(x, requires_grad_xyz=True)
                h = self.sigma_net(x_embed)
                sigma = trunc_exp(h[..., 0])
                
                lambda_ = 5e-2
                sigma_remap = 1 / lambda_ * (1 - torch.exp(-lambda_ * sigma))
                
                geo_feat = h[..., 1:]
                d_output = torch.ones_like(sigma, requires_grad=False)
                normal_grad = - torch.autograd.grad(
                    outputs=sigma_remap,
                    inputs=x,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                normal_grad = normal_grad / (normal_grad.norm(dim=-1, keepdim=True) + 1e-5)
                h_mask = torch.logical_and(h_mask, torch.logical_not(normal_grad.isnan()).all(dim=-1))
        else:
            if self.visual_mode == 'PhiF':
                x_embed, normal_coarse, normal_fine, h_mask, phi_embed = self.meshfea_field(x, return_phi_embed=True)
            elif self.visual_mode == 'theta' or self.visual_mode == 'phi' or self.visual_mode == 'Nl':
                x_embed, normal_coarse, normal_fine, h_mask, theta, phi = self.meshfea_field(x, return_rot_angles=True)
            else:
                x_embed, normal_coarse, normal_fine, h_mask = self.meshfea_field(x)
            h = self.sigma_net(x_embed)
            sigma = trunc_exp(h[..., 0])
            geo_feat = h[..., 1:]
            normal_grad = None
        
        ret_dict = {}
        if self.render_light_model:
            if self.training:
                if self.smooth_grad_weight != 0.:
                    # Avoid too noisy grad damaging the training :(
                    normal_grad_supervision = normal_grad.detach() * (1 - self.smooth_grad_weight) + normal_coarse * self.smooth_grad_weight
                    normal_grad_supervision = normal_grad_supervision / (normal_grad_supervision.norm(dim=-1, keepdim=True) + 1e-5)
                else:
                    normal_grad_supervision = normal_grad.detach()
                
                if composite_normal_error:
                    normal_grad_nan_mask = torch.logical_not(normal_grad_supervision.isnan().any(dim=-1, keepdim=True).float())
                    normal_error = 1e-1 * ((normal_fine - normal_grad_supervision) ** 2)
                    normal_error = torch.where(normal_grad_nan_mask.expand_as(normal_error), normal_error, torch.zeros_like(normal_error))
                    ret_dict['normal_error'] = normal_error
                else:
                    ret_dict['normal'] = normal_fine
                    ret_dict['normal_grad'] = normal_grad_supervision
            normal = normal_fine
        else:
            normal = normal_coarse

        # Normalize normals
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-5)
        normal_coarse = normal_coarse / (normal_coarse.norm(dim=-1, keepdim=True) + 1e-5)
        normal_coarse_rot = normal_coarse
        
        # Coarse and fine weighting
        if not self.training:
            normal = self.fc_weight * normal + (1 - self.fc_weight) * normal_coarse
            normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-5)

        # Euler rotation for changing lighting direction
        if euler is not None and not self.training:
            rot = torch.from_numpy(R.from_rotvec(euler).as_matrix()).to(d.dtype).to(d.device)
            d = torch.einsum('ab,nb->na', rot, d)
            normal = torch.einsum('ab,nb->na', rot, normal)
            normal_coarse_rot = torch.einsum('ab,nb->na', rot, normal_coarse)

        # color
        if self.training or self.visual_mode == 'RGB' or (not is_gui_mode):
            # RGB mode
            if not self.render_light_model:
                # Static light model
                if self.dir_degree > 0:
                    # Reflection
                    d_normalized = d / (d.norm(dim=-1, keepdim=True) + 1e-5)
                    wr = 2 * (-d_normalized * normal).sum(-1, keepdim=True) * normal + d_normalized
                    wr = (wr + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
                    dir_embed = self.encoder_dir(wr)
                    h = torch.cat([dir_embed, geo_feat], dim=-1)
                else:
                    h = geo_feat
                h = self.color_net(h)
                color = torch.sigmoid(h)
            else:
                # Light model
                if self.render_light_model:
                    # Spherical Gaussian and Spherical Harmonic Model
                    # normal_grad = normal_grad.detach()
                    normal = normal.detach() if self.training or not self.use_coarse_normal else normal_coarse_rot
                    normal = normal if self.training or not self.use_grad_normal else normal_grad.detach()
                    view_dirs = -d if self.light_model == 'SG' else d  # SG use -d instead of d
                    if self.coarse_as_primary:
                        normal_primary = normal_coarse_rot
                        normal_secondary = normal
                    else:
                        normal_primary = normal
                        normal_secondary = normal_coarse_rot
                    gamma = None if frame_index is None or not self.optimize_gamma else self.gammas[frame_index[0]]
                    full, specular, diffuse, albedo = self.light_net(geo_feat, normal_primary, view_dirs, normal_secondary=normal_secondary, shade_visibility=(not self.no_visibility) and (self.shade_visibility or self.training), gamma=gamma)
                    if self.training or self.light_visual_mode == 'Full':
                        color = full
                    elif self.light_visual_mode == 'Specular':
                        color = specular
                    elif self.light_visual_mode == 'Diffuse':
                        color = diffuse
                    elif self.light_visual_mode == 'Albedo':
                        color = albedo
                    else:
                        print('Unkown light mode: ', self.light_visual_mode)
                        exit(0)
                else:
                    print('Unknown light model: ', self.light_model)
                    exit(0)
        elif self.visual_mode == 'UV':
            # UVH mode
            uvh, h_mask_ = self.meshfea_field.uv(x)
            if uvh is not None:
                color = torch.zeros_like(x)
                color[..., :2] = uvh[..., :2]
                color = color.detach()
                sigma = sigma.detach()
                h_mask = h_mask_
        elif self.visual_mode == 'Tan' or self.visual_mode == 'Btan':
            # Tangent or Bitangent mode
            tbn = self.meshfea_field.tbn(x)
            color = tbn[:, 0] if self.visual_mode == 'Tan' else tbn[:, 1]
            color = ((color + 1) / 2).detach()
            sigma = sigma.detach()
        elif self.visual_mode == 'Grad':
            # Normal mode
            color = (normal_grad + 1) / 2
        elif self.visual_mode == 'Nc':
            color = (normal_coarse + 1) / 2
        elif self.visual_mode == 'Nf':
            color = (normal_fine + 1) / 2
        elif self.visual_mode == 'normal':
            color = (normal + 1) / 2
        elif self.visual_mode == 'Fea':
            if self.feature_visualize_func is None:
                sample_features = self.meshfea_field.sample_features().detach().cpu().numpy()
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                pca.fit(sample_features)
                feature_3vec = pca.transform(sample_features)
                self.feature_vis_bounds = torch.from_numpy(np.stack([feature_3vec.min(0), feature_3vec.std(0)], axis=0)).float().to(self.device)
                self.feature_vis_w = torch.from_numpy(pca.components_.T).float().to(self.device)
                self.feature_vis_mean = torch.from_numpy(sample_features.mean(0)).float().to(self.device)
                self.feature_visualize_func = lambda x: (torch.clamp(((x - self.feature_vis_mean) @ self.feature_vis_w) / self.feature_vis_bounds[1], -1, 1) + 1) / 2
            color = self.feature_visualize_func(x_embed[..., :self.meshfea_field.encoder_f_out_dim])
        else:
            print('Unkown visual mode id: ', self.visual_mode_id)
            exit(0)

        zero_color = torch.zeros_like(color)
        filtered_color = torch.where(h_mask.unsqueeze(-1), color, zero_color)
        zero_sigma = torch.zeros_like(sigma)
        filtered_sigma = torch.where(h_mask, sigma, zero_sigma)

        if self.optimize_gamma and frame_index is not None:
            gamma_loss = 5 * ((self.gammas[frame_index[0]]-2.4)**2).mean()
            ret_dict['gamma_loss'] = gamma_loss

        # uvh, h_mask = self.mfgrid.meshprojector.uvh(x, h_threshold=self.h_threshold)
        # color2 = uvh[..., :2]
        # filtered_color = torch.zeros_like(color)
        # filtered_sigma = h_mask.to(filtered_sigma.dtype) * 100
        return filtered_sigma, filtered_color, ret_dict
    

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x_embed, _, _, h_mask= self.meshfea_field(x)

        # sigma
        h = self.sigma_net(x_embed)
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        zero_sigma = torch.zeros_like(sigma)
        filtered_sigma = torch.where(h_mask, sigma, zero_sigma)

        # uvh, h_mask = self.mfgrid.meshprojector.uvh(x, h_threshold=self.h_threshold)
        # filtered_sigma = h_mask.to(filtered_sigma.dtype) * 100
        # filtered_sigma = torch.ones_like(h_mask.to(filtered_sigma.dtype)) * 100
        # geo_feat = None

        return {
            'sigma': filtered_sigma,
            'geo_feat': geo_feat,
        }

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.meshfea_field.parameters(), 'lr': lr}, 
        ]
        if self.render_light_model:
            params.append({'params': self.light_net.parameters(), 'lr': lr})
            if hasattr(self, 'normal_net') and self.normal_net is not None:
                params.append({'params': self.normal_net.parameters(), 'lr': lr})
        else:
            params.append({'params': self.color_net.parameters(), 'lr': lr})
        # if self.optimize_camera:
        #     params.append({'params': [self.dRs, self.dts, self.dfs], 'lr': lr})
        if self.optimize_camera:
            params.append({'params': [self.dfs], 'lr': lr})
        if self.optimize_gamma:
            params.append({'params': [self.gammas], 'lr': lr})
        
        return params

    def import_field(self, field_dict, fp16=True):
        features = field_dict[self.attribute_name]
        mesh = field_dict['mesh'][()]
        grid_gap = field_dict['grid_gap']
        sample_tbn, sample_tbn_ids = field_dict['sample_tbn'], field_dict['sample_tbn_ids']
        local_tbn = field_dict['local_tbn']
        phi_embed = field_dict['phi_embed']
        if mesh is None:
            H, W = features.shape[:2]
            x, y = np.meshgrid(np.linspace(-.5 * grid_gap * H, .5 * grid_gap * H, H, endpoint=True), np.linspace(-.5 * grid_gap * W, .5 * grid_gap * W, W, endpoint=True), indexing='ij')
            coor = np.stack([x, y, np.zeros_like(x)], axis=-1).reshape([-1, 3])
            mesh = trimesh.Trimesh(vertices=coor, faces=[])
            normals = np.zeros_like(coor)
            normals[..., -1] = 1.
            mesh.as_open3d.vertex_normals = o3d.utility.Vector3dVector(normals)
        self.meshfea_field.import_field(features.reshape([H, W, features.shape[-1]]), bounds=[.5 * grid_gap * H, .5 * grid_gap * W], sample_tbn=sample_tbn, sample_tbn_ids=sample_tbn_ids, local_tbn=local_tbn, phi_embed=phi_embed)  # , [grid_gap*H*.5, grid_gap*W*.5])
        print('Bounds: ', self.meshfea_field.bounds)
        self.initialize_states(fp16=fp16)
        print('Import Field Done!')
    
    def import_patch(self, field_dict, fp16=True):
        self.patch_id = self.patch_id % field_dict['patch_coors'].shape[0]
        patch_local_tbn = field_dict['patch_local_tbn'][self.patch_id].reshape([-1, 9])
        patch_coor = field_dict['patch_coors'][self.patch_id].reshape([-1, 3])
        patch_norm = field_dict['patch_norms'][self.patch_id]
        phi_embed = field_dict['patch_phi_embed'][self.patch_id].reshape([-1, field_dict['patch_phi_embed'][self.patch_id].shape[-1]]) if field_dict['patch_phi_embed'] is not None else None
        patch = field_dict['patches'][self.patch_id]
        mesh = trimesh.Trimesh(vertices=patch_coor, faces=[])
        normals = np.zeros_like(patch_coor)
        normals[:] = patch_norm
        mesh.as_open3d.vertex_normals = o3d.utility.Vector3dVector(normals)
        mesh.export('./test_data/patch.ply')
        self.meshfea_field.import_patch(features=patch.reshape([-1, patch.shape[-1]]), mesh=mesh, local_tbn=patch_local_tbn, phi_embed=phi_embed)
        self.initialize_states(fp16=fp16)
        self.patch_id += 1        
        print('Import Patch Done!')
    
    def import_shape(self, mesh_path, fp16=True):
        mesh = trimesh.load_mesh(mesh_path)
        mesh.vertices -= np.mean(mesh.vertices, axis=0)
        mesh.vertices /= (np.absolute(mesh.vertices).max() + 1e-5)
        mesh.vertices /= 1.2
        self.meshfea_field.import_shape(mesh)
        self.initialize_states(fp16=fp16)
        print('Imported Shape Done!')
    
    def import_unhash(self, data_path, fp16=True):
        self.meshfea_field.import_unhash(data_path=data_path)
        self.initialize_states(fp16=fp16)
        print('Import Unhask Done!')
    
    def switch_shape_feature(self, fp16=True):
        if self.meshfea_field.imported_type == 'field':
            self.meshfea_field.imported_type = 'shape'
        elif self.meshfea_field.imported_type == 'shape':
            self.meshfea_field.imported_type = 'field'
        else:
            print('Nothing imported')
            return
        self.initialize_states(fp16=fp16)
        print('Import Unhask Done!')
    
    def switch_import(self, fp16=True):
        self.meshfea_field.imported = not self.meshfea_field.imported
        self.initialize_states(fp16=fp16)
    
    def set_uv_utilize_rate(self, rate=1., fp16=True):
        self.meshfea_field.uv_utilize_rate = rate
        self.initialize_states(fp16=fp16)
    
    def set_k_for_uv(self, k_for_uv=5, fp16=True):
        self.meshfea_field.K_for_uv = k_for_uv
        self.initialize_states(fp16=fp16)
    
    def set_sdf_factor(self, sdf_factor=1., fp16=True):
        self.meshfea_field.sdf_scale_factor = sdf_factor
        self.initialize_states(fp16=fp16)
    
    def set_sdf_offset(self, sdf_offset=0., fp16=True):
        self.meshfea_field.sdf_offset = sdf_offset
        self.initialize_states(fp16=fp16)
    
    def set_h_threshold(self, h_threshold=0.03, fp16=True):
        self.meshfea_field.h_threshold = h_threshold
        self.initialize_states(fp16=fp16)
    
    def initialize_states(self, fp16):
        with torch.cuda.amp.autocast(enabled=fp16):
            for _ in range(50):
                self.update_extra_state(decay=1., force_full_update=True, force_full_grid=True)
    
    def export_field(self, scan_pcl_path=None, picked_faces_path=None, record_rgb=False, work_space='./test_data/'):
        patches, grid_gap, patch_coors, patch_norms, patch_sample_tbn, patch_local_tbn, picked_vertices, patch_phi_embed, patch_rays = self.meshfea_field.sample_patches(scan_pcl_path=scan_pcl_path, picked_faces_path=picked_faces_path, record_rays=record_rgb, work_space=work_space)
        return {'mesh': trimesh.load_mesh(self.meshfea_field.mesh_path), 'patches': patches, 'grid_gap': grid_gap, 'patch_coors': patch_coors, 'patch_norms': patch_norms, 'patch_sample_tbn': patch_sample_tbn, 'patch_local_tbn': patch_local_tbn, 'picked_vertices': picked_vertices, 'patch_phi_embed': patch_phi_embed, 'patch_rays': patch_rays}

    def visualize_features(self,sv_path='./'):
        self.meshfea_field.visualize_features(sv_path=sv_path)
    
    def update_gridfield(self, target_stage=None):
        return self.meshfea_field.update(target_level=target_stage)

def FClayers(in_dim, out_dim, n_neurons=64, num_layers=3):
    layers = []
    dim = in_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(dim, n_neurons))
        layers.append(nn.ReLU(True))
        dim = n_neurons
    layers.append(nn.Linear(dim, out_dim))
    layer = nn.Sequential(*layers)
    return layer
