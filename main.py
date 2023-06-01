import torch
import argparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *
from tools.shape_tools import *
import shutil


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='')
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=40000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--mode', type=str, default='colmap', help="dataset mode, supports (colmap, blender)")
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0., help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()

    opt.O = True
    opt.bound = 1.0
    opt.scale = 0.8
    opt.dt_gamma = 0
    opt.mode = 'colmap'
    opt.gui = True
    opt.lr = 1e-2
    # opt.W = 192*4
    # opt.H = 108*4

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True

    surface_type = 'coacd_remesh'
    data_type = ''
    hull_num = 1
    dir_degree = 6
    hash = True
    normalize = True
    regularization = True
    clustering = True
    prob_model = False
    torch_sigma_layer = False
    light_model = 'SH'
    optimize_camera = True
    optimize_gamma = False
    lip_mlp = False
    coacd_threshold = .5
    pattern_rate = 1 / 50
    num_level = 8
    no_visibility = False
    bound_output_normal = False

    from data_args import *
    opt.path = PATH_TO_DATASET + '/' + DATA_NAME
    opt.workspace = './logs/' + data_type + '/' + DATA_NAME

    opt.preload = False

    from nerf.network_curvedfield import NeRFNetwork

    print(opt)
    
    seed_everything(opt.seed)

    surface_mesh_path = opt.workspace + '/meshes/surface_' + surface_type + '.obj'
    template_path='./data/template/src_model.obj'
    if not os.path.exists(surface_mesh_path):
        if surface_type == 'dach_reg' or surface_type == 'coacd_reg':
            ply_name = sorted([x for x in os.listdir(opt.workspace + '/meshes/') if x.startswith('ngp_') and x.endswith('.obj')])[-1]
            ply_path = opt.workspace + '/meshes/' + ply_name
            decomp_path = DACH(mesh_path=ply_path, hull_num=hull_num) if surface_type == 'dach_remesh' else CoACD(mesh_path=ply_path, threshold=coacd_threshold)
            union_path = MeshUnion(mesh_path=decomp_path) if surface_type == 'dach_remesh' else MeshUnion_manifold(mesh_path=decomp_path)
            smooth_path = Smooth(mesh_path=union_path)
            if 'coacd' in surface_type:
                smooth_path = Align(smooth_path, ply_path)
            result_path = Register(src_path=template_path, trg_path=smooth_path, save2trg=True)
        elif surface_type == 'dach_remesh' or surface_type == 'coacd_remesh':
            ply_name = sorted([x for x in os.listdir(opt.workspace + '/meshes/') if x.startswith('ngp_') and x.endswith('.obj')])[-1]
            ply_path = opt.workspace + '/meshes/' + ply_name
            decomp_path = DACH(mesh_path=ply_path, hull_num=hull_num) if surface_type == 'dach_remesh' else CoACD(mesh_path=ply_path, threshold=coacd_threshold)
            union_path = MeshUnion(mesh_path=decomp_path) if surface_type == 'dach_remesh' else MeshUnion_manifold(mesh_path=decomp_path)
            smooth_path = Smooth(mesh_path=union_path)
            if 'coacd' in surface_type:
                smooth_path = Align(smooth_path, ply_path)
            union_mesh = pymesh.load_mesh(smooth_path)
            result_mesh = remesh(union_mesh, 'normal')
            result_path = surface_mesh_path
            pymesh.save_mesh(result_path, result_mesh)
        elif surface_type == 'pcl_reg':
            ply_path = opt.workspace + '/meshes/pcl.ply'
            result_path = Register(src_path=template_path, trg_path=ply_path, save2trg=True, trg_is_ply=True)
        else:
            print('Unkown surface type: ', surface_type)
            exit(0)
        shutil.move(result_path, surface_mesh_path)
    surface_mesh = trimesh.load_mesh(surface_mesh_path)
    ply_path = opt.workspace + '/meshes/pcl.ply'
    if not os.path.exists(opt.workspace + '/meshes/h_threshold.npz'):
        print('Calculating H threshold...')
        pcd = o3d.io.read_point_cloud(ply_path)
        pcd = pcd.voxel_down_sample(voxel_size=0.01)
        scanned_ply = np.asarray(pcd.points)
        udf = np.abs(trimesh.proximity.ProximityQuery(surface_mesh).signed_distance(scanned_ply))
        udf_07 = np.partition(udf, -int(udf.shape[0] * .3))[-int(udf.shape[0] * .3)]
        h_threshold = 2 * udf_07
        np.savez(opt.workspace + '/meshes/h_threshold', h_threshold=h_threshold)
    else:
        h_threshold = float(np.load(opt.workspace + '/meshes/h_threshold.npz')['h_threshold'])
    print('H threshold thickness: ', h_threshold)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = NeRFDataset(opt, device=device, type='trainval', normalize=normalize, optimize_camera=optimize_camera).dataloader()

    model = NeRFNetwork(
        surface_mesh_path=surface_mesh_path,
        h_threshold=h_threshold,
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        dir_degree=dir_degree,
        bg_radius=opt.bg_radius,
        hash=hash,
        clustering=clustering,
        prob_model=prob_model,
        torch_sigma_layer=torch_sigma_layer,
        light_model=light_model,
        num_level=num_level,
        regularization=regularization,
        cal_dist_loss=False,  # The effect of dist loss becomes wierd after the lib got upgraded!!!!!!!!
        optimize_camera=optimize_camera,
        camera_num=train_loader._data.length,
        optimize_gamma=optimize_gamma,
        lip_mlp=lip_mlp,
        pattern_rate=pattern_rate,
        no_visibility=no_visibility,
        bound_output_normal=bound_output_normal,
    )
    
    print(model)
    criterion = torch.nn.L1Loss()
    optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
    scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
    trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=[PSNRMeter()], use_checkpoint=opt.ckpt, eval_interval=50)
    trainer.train_loader = train_loader # attach dataloader to trainer
    gui = NeRFGUI(opt, trainer, gui_mode=opt.gui)
    gui.render()
