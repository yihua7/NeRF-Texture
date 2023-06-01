import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.encoding import get_encoder
from tools.activation import trunc_exp
from ffmlp import FFMLP

from .renderer import NeRFRenderer

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        self.sigma_net = FFMLP(
            input_dim=self.in_dim, 
            output_dim=1 + self.geo_feat_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir)
        self.in_dim_color += self.geo_feat_dim + 1 # a manual fixing to make it 32, as done in nerf_network.h#178
        
        self.color_net = FFMLP(
            input_dim=self.in_dim_color, 
            output_dim=3,
            hidden_dim=self.hidden_dim_color,
            num_layers=self.num_layers_color,
        )

        self.visual_modes = ['RGB', 'Gradient']
        self.visual_mode_id = 0
        self.visual_mode = self.visual_modes[self.visual_mode_id]
    
    def forward(self, x, d, is_gui_mode=False, **kwargs):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]


        if (not self.training) and self.visual_mode == 'Gradient' and is_gui_mode:
            # Normal mode
            with torch.enable_grad():
                x.requires_grad_(True)
                x_ = self.encoder(x, bound=self.bound)
                h = self.sigma_net(x_, force_grad=True)
                #sigma = F.relu(h[..., 0])
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
        else:
            # sigma
            x = self.encoder(x, bound=self.bound)
            h = self.sigma_net(x)
            #sigma = F.relu(h[..., 0])
            sigma = trunc_exp(h[..., 0])
        
        geo_feat = h[..., 1:]
        # color        
        d = self.encoder_dir(d)

        # TODO: preallocate space and avoid this cat?
        p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat, p], dim=-1)
        h = self.color_net(h)
        
        if self.training or self.visual_mode == 'RGB':
            # sigmoid activation for rgb
            rgb = torch.sigmoid(h)
        elif self.visual_mode == 'Gradient':
            # Normal mode
            rgb = (normal_grad + 1) / 2
        
        return sigma, rgb, {}

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = self.encoder(x, bound=self.bound)
        h = self.sigma_net(x)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def random_density(self, pnum=1024, dtype=torch.float16, device=torch.device("cuda:0")):
        x = torch.rand([pnum, 3], dtype=dtype, device=device)
        density = self.density(x)
        return density['sigma']

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        #starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        #starter.record()

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

            #print(x.shape, rgbs.shape)

        #ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f'mask = {curr_time}')
        #starter.record()

        d = self.encoder_dir(d)

        p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat, p], dim=-1)

        h = self.color_net(h)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        #ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f'call = {curr_time}')
        #starter.record()

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
        else:
            rgbs = h

        #ender.record(); torch.cuda.synchronize(); curr_time = starter.elapsed_time(ender); print(f'unmask = {curr_time}')
        #starter.record()

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        if self.optimize_camera:
            params.append({'params': [self.dRs, self.dts, self.dfs], 'lr': lr})
        # if self.optimize_camera:
        #     params.append({'params': [self.dfs], 'lr': lr})
        if self.optimize_gamma:
            params.append({'params': [self.gammas], 'lr': lr})
        
        return params