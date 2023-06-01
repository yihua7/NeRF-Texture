import os
import copy
import scipy
import torch
import shutil
import pymesh
import mcubes
import trimesh
import subprocess
import numpy as np
import open3d as o3d
from tqdm import tqdm
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from plyfile import PlyElement, PlyData
from sklearn.neighbors import NearestNeighbors
# from pytorch3d.io import load_obj, save_obj
# from pytorch3d.structures import Meshes
# from pytorch3d.utils import ico_sphere
# from pytorch3d.ops import sample_points_from_meshes
# from pytorch3d.loss import (
#     chamfer_distance, 
#     mesh_edge_loss, 
#     mesh_laplacian_smoothing, 
#     mesh_normal_consistency,
# )


def remesh(mesh, detail="normal"):
    """Fix mesh to a uniform density"""
    print('Fixing mesh in ', detail, ' mode...')
    bbox_min, bbox_max = mesh.bbox
    diag_len = norm(bbox_max - bbox_min)
    if detail == "normal":
        target_len = diag_len * 5e-3
    elif detail == "high":
        target_len = diag_len * 2.5e-3
    elif detail == "low":
        target_len = diag_len * 8e-3
    print("Target resolution: {} mm".format(target_len))

    count = 0
    mesh, _ = pymesh.remove_duplicated_vertices(mesh, 1e-3)
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100)
    mesh, __ = pymesh.split_long_edges(mesh, target_len)
    num_vertices = mesh.num_vertices
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                                               preserve_feature=True)
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print("#v: {}".format(num_vertices))
        count += 1
        if count > 10: break

    mesh = pymesh.resolve_self_intersection(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh = pymesh.compute_outer_hull(mesh)
    mesh, __ = pymesh.remove_duplicated_faces(mesh)
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh


def pca_plane(mesh_path):
    vertices = trimesh.load_mesh(mesh_path).vertices
    pca = PCA(n_components=3)
    pca.fit(vertices)
    components = pca.components_
    
    # To right hand system
    xaxis, yaxis, zaxis = components
    if np.dot(zaxis, np.array([0, 1, 0])) < 0:
        zaxis = - zaxis  # zaxis should be consistent to y axis from COLMAP
    xaxis = np.cross(yaxis, zaxis)
    components = np.stack([xaxis, yaxis, zaxis], axis=0)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = components
    transform[:3, 3] = - components @ vertices.mean(0)
    vertices = np.concatenate([vertices, np.ones_like(vertices[..., :1])], axis=-1)
    vertices = vertices @ transform.T
    vertices = vertices[..., :3]
    return transform, vertices


def mesh_edit_example():
    style_path = './style_nerf_lego_gui/meshes/dolphin.obj'
    style_mesh = pymesh.load_mesh(style_path)
    vn = np.array(style_mesh.vertices)
    vn[..., 0] *= -1
    pymesh.save_mesh(style_path, pymesh.form_mesh(vertices=vn, faces=style_mesh.faces))


def write_ply(v, path):
    header = f"ply\nformat ascii 1.0\nelement vertex {len(v)}\nproperty double x\nproperty double y\nproperty double z\nend_header\n"
    str_v = [f"{vv[0]} {vv[1]} {vv[2]}\n" for vv in v]

    with open(path, 'w') as meshfile:
        meshfile.write(f'{header}{"".join(str_v)}')


def write_ply_rgb(points, RGB, filename):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as PLY file """
    N = points.shape[0]
    vertex = []
    for i in range(N):
        vertex.append((points[i, 0], points[i, 1], points[i, 2], RGB[i][0], RGB[i][1], RGB[i][2]))
    vertex = np.array(vertex,
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)


def remove_isolated_piecies(mesh, min_face_num=100):
    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=min_face_num)
    mask = np.zeros(len(mesh.faces), dtype=np.bool)
    mask[np.concatenate(cc)] = True
    mesh.update_faces(mask)
    return mesh


def DACH(mesh_path, hull_num=50, force=True):
    """Decomposed Approximated Convex Hulls"""
    mesh_name = mesh_path.split('/')[-1].split('.')[0]
    path = '/'.join(mesh_path.split('/')[:-1])
    save_path = path + '/' + mesh_name + '_dach' + str(hull_num) + '.obj'
    if os.path.exists(save_path) and not force:
        return save_path
    print('Decomposing ' + mesh_path + ' into Approximated Convex Hulls')
    cmd = 'testVHACD ' + mesh_path + ' -h ' + str(hull_num)
    subprocess.call(cmd.split(' '), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    shutil.move('decomp.obj', save_path)
    print('Done with DACH! Saved in ' + save_path)
    return save_path


def CoACD(mesh_path, threshold=.5, force=True):
    """Decomposed Approximated Convex Hulls (SIGGRAPH 2022)"""
    mesh_name = mesh_path.split('/')[-1].split('.')[0]
    path = '/'.join(mesh_path.split('/')[:-1])
    save_path = path + '/' + mesh_name + '_coacd.obj'
    if os.path.exists(save_path) and not force:
        return save_path
    print('Decomposing ' + mesh_path + ' into Approximated Convex Hulls')
    cmd = './tools/CoACD ' + f'-i {mesh_path} -o {save_path} --pca' + f' -t {threshold}'
    subprocess.call(cmd.split(' '), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print('Done with CoACD! Saved in ' + save_path)
    return save_path


def Convert2Tetrahedron(mesh_path):
    print('Converting' + mesh_path + ' to tetrahedral mesh ..')
    cmd = 'TetWild ' + mesh_path
    path = '/'.join(mesh_path.split('/')[:-1])
    mesh_name = mesh_path.split('/')[-1].split('.')[0]
    save_path = path + '/' + mesh_name + '_.msh'
    subprocess.call(cmd.split(' '), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print('Conversion done! Saved in ' + save_path)
    return save_path


def MeshUnion_marching_cube(mesh_path, resolution=256, do_dilation=False):
    print('Calculate the unoin of watertight parts of ' + mesh_path + ' with resolution ' + str(resolution) + ' for marching cube.')
    mesh = trimesh.load(mesh_path)
    bound_min, bound_max = mesh.vertices.min(), mesh.vertices.max()
    bound_center, rescale = (bound_min + bound_max) / 2, 1.2
    bound_min = rescale * (bound_min - bound_center) + bound_center
    bound_max = rescale * (bound_max - bound_center) + bound_center

    N = 64
    resolution = 128
    X = np.linspace(bound_min, bound_max, resolution)
    Y = np.linspace(bound_min, bound_max, resolution)
    Z = np.linspace(bound_min, bound_max, resolution)
    xx, yy = np.meshgrid(X, Y, indexing='ij')
    pts = np.stack([xx.reshape([-1]), yy.reshape([-1])], axis=-1)
    pts = np.concatenate([pts, bound_min * np.ones_like(pts[..., :1])], axis=-1)
    ray_origins = pts
    ray_directions = np.zeros_like(ray_origins)
    ray_directions[..., -1] = 1
    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)

    # Split water tight meshes
    meshes = mesh.split(only_watertight=True)

    for mesh in meshes:
        # run the mesh- ray test
        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions)
        for ray_idx in np.unique(index_ray):
            idx = np.where(index_ray == ray_idx)[0]
            location = locations[idx]
            if location.shape[0] < 2:
                continue
            zmin, zmax = sorted(location[..., -1])
            mask = np.logical_and(Z > zmin, Z < zmax)
            u[ray_idx // resolution, ray_idx % resolution][mask] = 1
    
    if do_dilation:
        kernel_size = 5
        u = scipy.ndimage.morphology.grey_dilation(u, size=(kernel_size, kernel_size, kernel_size))
    
    vertices, triangles = mcubes.marching_cubes(u, 1e-5)
    vertices = vertices / (resolution - 1.0) * (bound_max - bound_min) + bound_min
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    path = '/'.join(mesh_path.split('/')[:-1])
    mesh_name = mesh_path.split('/')[-1].split('.')[0]
    save_path = path + '/' + mesh_name + '_union.obj'
    mesh.export(save_path)
    print('Done with union calculation! Saved in ' + save_path)
    return save_path


def MeshSimplify(mesh_path, preserve_feature=True):
    print('Symplifying mesh: ' + mesh_path)
    mesh = pymesh.load_mesh(mesh_path)
    edge_num = int(1.5 * mesh.faces.shape[0])
    trg_edge_num = 120
    if edge_num <= trg_edge_num:
        print('No need to symplify. Edge number: ', edge_num)
        return mesh_path
    rel_threshold = (edge_num - trg_edge_num) / edge_num
    new_mesh, info = pymesh.collapse_short_edges(mesh, rel_threshold=rel_threshold, preserve_feature=preserve_feature)
    path = '/'.join(mesh_path.split('/')[:-1])
    mesh_name = mesh_path.split('/')[-1].split('.')[0]
    save_path = path + '/' + mesh_name + '_simp.obj'
    pymesh.save_mesh(save_path, new_mesh)
    print('Down with simplification. Saved in ' + save_path)
    return save_path


def MeshUnion(mesh_path):
    print('Calculate the unoin of watertight parts of ' + mesh_path)
    mesh = trimesh.load(mesh_path)
    # Split water tight meshes
    meshes = mesh.split(only_watertight=True)

    meshes_pymesh = [pymesh.form_mesh(meshes[i].vertices, meshes[i].faces) for i in range(len(meshes))]
    meshes_union = None
    for i in range(len(meshes_pymesh)):
        if meshes_union is None:
            meshes_union = meshes_pymesh[i]
        else:
            meshes_union = pymesh.boolean(meshes_union, meshes_pymesh[i], operation="union")
    path = '/'.join(mesh_path.split('/')[:-1])
    mesh_name = mesh_path.split('/')[-1].split('.')[0]
    save_path = path + '/' + mesh_name + '_union.obj'
    pymesh.save_mesh(save_path, meshes_union) 
    print('Done with union calculation! Saved in ' + save_path)
    return save_path


def MeshUnion_manifold(mesh_path):
    print('Calculate the unoin manifold of watertight parts of ' + mesh_path)
    mesh_name = mesh_path.split('/')[-1].split('.')[0]
    path = '/'.join(mesh_path.split('/')[:-1])
    save_path = path + '/' + mesh_name + '_union.obj'
    cmd = './tools/manifold ' + f' {mesh_path} {save_path}'
    os.system(cmd)
    print('Done with union manifold calculation! Saved in ' + save_path)
    return save_path


def Register(src_path, trg_path, save2trg=True, force=True, trg_is_ply=False):

    path = '/'.join(trg_path.split('/')[:-1]) if save2trg else '/'.join(src_path.split('/')[:-1])
    src_name = src_path.split('/')[-1].split('.')[0]
    trg_name = trg_path.split('/')[-1].split('.')[0]
    save_path = path + '/' + src_name + '_reg2_' + trg_name + '.obj'

    if os.path.exists(save_path) and not force:
        return save_path

    print('Registering ' + src_path + ' to ' + trg_path + ' with pytorch3d ...')

    # Set the device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    def normalize_vertices(verts):
        center = (verts.max(0)[0] + verts.min(0)[0]) / 2
        verts = verts - center
        scale = max(verts.abs().max(0)[0])
        verts = verts / scale
        return verts, scale, center

    if trg_is_ply:
        trg_verts = trimesh.load_mesh(trg_path).vertices
        trg_verts = torch.from_numpy(trg_verts).to(device)
        trg_verts = trg_verts.to(device)
        trg_verts, trg_scale, trg_center = normalize_vertices(trg_verts)
    else:
        trg_verts, trg_faces, _ = load_obj(trg_path)
        trg_faces_idx = trg_faces.verts_idx.to(device)
        trg_verts = trg_verts.to(device)
        trg_verts, trg_scale, trg_center = normalize_vertices(trg_verts)
        trg_mesh = Meshes(verts=[trg_verts], faces=[trg_faces_idx])

    src_verts, src_faces, _ = load_obj(src_path)
    src_faces_idx = src_faces.verts_idx.to(device)
    src_verts = src_verts.to(device)
    src_verts, _, _ = normalize_vertices(src_verts)
    src_mesh = Meshes(verts=[src_verts], faces=[src_faces_idx])
    src_faces_idx = src_faces.verts_idx.to(device)

    deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([deform_verts], lr=1.0, momentum=0.9)

    # Number of optimization steps
    Niter = 2000
    # Weight for the chamfer loss
    w_chamfer = 1.0 
    # Weight for mesh edge loss
    w_edge = 1.0 
    # Weight for mesh normal consistency
    w_normal = 0.01 
    # Weight for mesh laplacian smoothing
    w_laplacian = 0.5
    # Plot period for the losses
    plot_period = 250
    loop = tqdm(range(Niter))

    chamfer_losses = []
    laplacian_losses = []
    edge_losses = []
    normal_losses = []


    for _ in loop:
        # Initialize optimizer
        optimizer.zero_grad()
        
        # Deform the mesh
        new_src_mesh = src_mesh.offset_verts(deform_verts)
        
        # We sample 5k points from the surface of each mesh 
        sample_src = sample_points_from_meshes(new_src_mesh, 5000)
        if trg_is_ply:
            replace = trg_verts.shape[0] < 5000
            sample_trg = trg_verts[torch.from_numpy(np.random.choice(np.arange(trg_verts.shape[0]), [5000], replace=replace))]
            sample_trg = sample_trg.reshape(sample_src.shape).to(sample_src.dtype)
        else:
            sample_trg = sample_points_from_meshes(trg_mesh, 5000)
        
        # We compare the two sets of pointclouds by computing (a) the chamfer loss
        loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)
        # and (b) the edge length of the predicted mesh
        loss_edge = mesh_edge_loss(new_src_mesh)
        # mesh normal consistency
        loss_normal = mesh_normal_consistency(new_src_mesh)
        # mesh laplacian smoothing
        loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
        # Weighted sum of the losses
        loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
        # Print the losses
        loop.set_description('total_loss = %.6f' % loss)
        
        # Save the losses for plotting
        chamfer_losses.append(float(loss_chamfer.detach().cpu()))
        edge_losses.append(float(loss_edge.detach().cpu()))
        normal_losses.append(float(loss_normal.detach().cpu()))
        laplacian_losses.append(float(loss_laplacian.detach().cpu()))
            
        # Optimization step
        loss.backward()
        optimizer.step()

    fig = plt.figure(figsize=(13, 5))
    ax = fig.gca()
    ax.plot(chamfer_losses, label="chamfer loss")
    ax.plot(edge_losses, label="edge loss")
    ax.plot(normal_losses, label="normal loss")
    ax.plot(laplacian_losses, label="laplacian loss")
    ax.legend(fontsize="16")
    ax.set_xlabel("Iteration", fontsize="16")
    ax.set_ylabel("Loss", fontsize="16")
    ax.set_title("Loss vs iterations", fontsize="16")
    fig.savefig(path + src_name + '_reg2_' + trg_name  + '.png')

    # Fetch the verts and faces of the final predicted mesh
    final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)
    # Scale normalize back to the original target size
    final_verts = final_verts * trg_scale + trg_center
    # Store the predicted mesh using save_obj
    save_obj(save_path, final_verts, final_faces)
    print('Registeration done! Saved in ' + save_path)
    return save_path


def ARAP_deform(msh_path, src_template_path, trg_template_path, tmp_trg_scale=1., sv_path=None):

    path = '/'.join(msh_path.split('/')[:-1])
    mesh_name = msh_path.split('/')[-1].split('.')[0]
    src_name = src_template_path.split('/')[-1].split('.')[0]
    trg_name = trg_template_path.split('/')[-1].split('.')[0]
    save_path = path + '/' + mesh_name + '_' + src_name + '2' + trg_name + 'arap.obj' if sv_path is None else sv_path + '/' + mesh_name + '_' + src_name + '2' + trg_name + 'arap.obj'
    save_path_msh = path + '/' + mesh_name + '_' + src_name + '2' + trg_name + 'arap.msh' if sv_path is None else sv_path + '/' + mesh_name + '_' + src_name + '2' + trg_name + 'arap.msh'

    print('ARAP deforming ...')
    mesh_pm = pymesh.meshio.load_mesh(msh_path)
    vertices = mesh_pm.vertices
    voxels = mesh_pm.voxels
    faces = []
    for i in range(voxels.shape[0]):
        voxel = np.sort(voxels[i])
        faces.append(voxel[[0, 1, 2]])
        faces.append(voxel[[0, 1, 3]])
        faces.append(voxel[[0, 2, 3]])
        faces.append(voxel[[1, 2, 3]])
    faces = np.stack(faces, axis=0)
    faces = np.unique(faces, axis=0)

    surfaces = mesh_pm.faces
    surface_vert_idx = np.unique(surfaces.reshape([-1]))
    surface_vert = vertices[surface_vert_idx]
    ##############################
    src_vertices = vertices
    src_faces = faces
    ##############################
    src_mesh = trimesh.Trimesh(vertices, src_faces, process=False)
    src_mesh.export(path + '/' + mesh_name + '_tet.obj')

    src_template_mesh = pymesh.load_mesh(src_template_path)
    src_template_vert = src_template_mesh.vertices
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(src_template_vert)
    _, nn_indices = nbrs.kneighbors(surface_vert)
    ##############################
    nn_indices = nn_indices.squeeze()
    ##############################

    trg_template_mesh = pymesh.load_mesh(trg_template_path)
    ##############################
    src_center = (src_template_vert.max(0) + src_template_vert.min(0)) / 2
    src_scale = np.abs(src_template_vert - src_center).max()
    trg_template_vert = trg_template_mesh.vertices
    trg_center = (trg_template_vert.max(0) + trg_template_vert.min(0)) / 2
    trg_template_vert = trg_template_vert - trg_center
    trg_scale = np.abs(trg_template_vert).max()
    trg_template_vert = trg_template_vert / trg_scale
    trg_template_vert = trg_template_vert * src_scale * tmp_trg_scale + src_center
    ##############################

    v, f = src_vertices, src_faces
    # Vertices in selection
    b = surface_vert_idx.reshape([-1, 1])
    # Precomputation
    import igl
    arap = igl.ARAP(v, f, 3, b)
    # Plot the mesh with pseudocolors
    bc = trg_template_vert[nn_indices]
    vn = arap.solve(bc, v)
    ##############################
    vn = vn - src_center
    vn = vn / tmp_trg_scale / src_scale
    vn = vn * trg_scale + trg_center
    ##############################

    mesh = trimesh.Trimesh(vn, f, process=False)
    mesh.export(save_path)

    mesh_pm_arap = pymesh.form_mesh(vertices=vn, faces=mesh_pm.faces, voxels=mesh_pm.voxels)
    pymesh.meshio.save_mesh(save_path_msh, mesh_pm_arap)
    print('ARAP deformation done! Saved to ' + save_path_msh)
    return save_path_msh, save_path


def Align(src_path, trg_path):
    print('Models aligning ...')
    path = '/'.join(src_path.split('/')[:-1])
    src_name = src_path.split('/')[-1].split('.')[0]
    trg_name = trg_path.split('/')[-1].split('.')[0]
    format = src_path.split('./')[-1].split('.')[-1]
    save_path = path + '/' + src_name + '_align2_' + trg_name + '.' + format

    src_mesh, trg_mesh = pymesh.load_mesh(src_path), pymesh.load_mesh(trg_path)
    src_vert, trg_vert = src_mesh.vertices, trg_mesh.vertices

    src_center = (src_vert.max(0) + src_vert.min(0)) / 2
    src_scale = np.abs(src_vert - src_center).max()

    trg_center = (trg_vert.max(0) + trg_vert.min(0)) / 2
    trg_scale = np.abs(trg_vert - trg_center).max()

    src_vert_new = (np.array(src_vert) - src_center) / src_scale * trg_scale + trg_center
    if format == 'msh':
        pymesh.save_mesh(save_path, pymesh.form_mesh(vertices=src_vert_new, faces=src_mesh.faces, voxels=src_mesh.voxels))
    else:
        pymesh.save_mesh(save_path, pymesh.form_mesh(vertices=src_vert_new, faces=src_mesh.faces))
    print('Alignment done! Saved to ' +  save_path)
    return save_path


def Smooth(mesh_path):
    print('Smoothing ...')
    mesh_name = mesh_path.split('/')[-1].split('.')[0]
    path = '/'.join(mesh_path.split('/')[:-1])
    save_path = path + '/' + mesh_name + '_smooth.obj'
    new_mesh = trimesh.smoothing.filter_laplacian(trimesh.load_mesh(mesh_path), iterations=5)
    new_mesh.export(save_path)
    return save_path


def ICP(src_ply, trg_ply, p2p=True):
    def draw_registration_result(source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])

    threshold = 0.02
    trans_init = np.eye(4)
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(src_ply)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(trg_ply)

    if p2p:
        print("Apply point-to-point ICP")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)
        print("")
        draw_registration_result(source, target, reg_p2p.transformation)
        source.transform(reg_p2p.transformation)
        reg_src_ply = np.asarray(source.points)
        return reg_p2p.transformation, reg_src_ply
    else:
        print("Apply point-to-plane ICP")
        reg_p2l = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        print(reg_p2l)
        print("Transformation is:")
        print(reg_p2l.transformation)
        print("")
        draw_registration_result(source, target, reg_p2l.transformation)
        source.transform(reg_p2l.transformation)
        reg_src_ply = np.asarray(source.points)
        return reg_p2l.transformation, reg_src_ply


def Expand(mesh_path, scale=1.5):
    print('Expanding mesh by scale: ', scale)
    mesh = trimesh.load_mesh(mesh_path)
    center = (mesh.vertices.max(axis=0) + mesh.vertices.min(axis=0)) / 2
    mesh.vertices = scale * (mesh.vertices - center) + center
    
    path = '/'.join(mesh_path.split('/')[:-1])
    mesh_name = mesh_path.split('/')[-1].split('.')[0]
    save_path = path + '/' + mesh_name + '_expand.obj'
    mesh.export(save_path)
    print('Expanding done. Saved in ' + save_path)
    return save_path


def Direct_deform(src_path, style_path, hull_num=3):
    # 1. Approximate source mesh with decomposed convex hulls
    src_decomp_path = DACH(mesh_path=src_path, hull_num=hull_num)
    # 2. Union of source convex hulls
    src_union_path = MeshUnion_marching_cube(mesh_path=src_decomp_path, resolution=256, do_dilation=True)
    # 3. Convert source hull mesh into tetrahedron
    src_msh_path = Convert2Tetrahedron(mesh_path=src_union_path)
    
    # 4. Approximate style mesh with decomposed convex hulls
    style_decomp_path = DACH(mesh_path=style_path, hull_num=hull_num)
    # 5. Union of style convex hulls
    style_union_path = MeshUnion_marching_cube(mesh_path=style_decomp_path, resolution=256, do_dilation=True)
    
    # 7. Register source hull mesh to style hull mesh
    reg_src_path = Register(src_path=src_union_path, trg_path=style_union_path)
    # 8. Deform source tetrahedron with source hull mesh controling points
    msh_arap_src_path, msh_arap_src_path_obj = ARAP_deform(msh_path=src_msh_path, src_template_path=src_union_path, trg_template_path=reg_src_path, tmp_trg_scale=1.)

    # 9. Align arap deformed source msh to original source msh
    msh_arap_src_path_align = Align(src_path=msh_arap_src_path, trg_path=src_msh_path)
    Align(src_path=msh_arap_src_path_obj, trg_path=src_msh_path)
    print('Stylized MSH: ' + msh_arap_src_path_align, '\nContent MSH: ' + src_msh_path)


def InDirect_deform(src_path, style_path, template_path='./data/template/src_model.obj'):
    # 1. Approximate source mesh with decomposed convex hulls
    src_decomp_path = DACH(mesh_path=src_path)
    # 2. Union of source convex hulls
    # src_union_path = MeshUnion(mesh_path=src_decomp_path)
    src_union_path = MeshUnion_marching_cube(mesh_path=src_decomp_path, resolution=256, do_dilation=True)
    # 3. Register template hull mesh to source hull mesh
    reg_src_path = Register(src_path=template_path, trg_path=src_union_path, save2trg=True)
    # 4. Convert template template hull mesh into tetrahedron
    template_msh_path = Convert2Tetrahedron(mesh_path=template_path)
    # 5. Deform template template tetrahedron with source hull mesh controling points
    msh_arap_src_path, msh_arap_src_path_obj = ARAP_deform(msh_path=template_msh_path, src_template_path=template_path, trg_template_path=reg_src_path, tmp_trg_scale=2., sv_path='/'.join(reg_src_path.split('/')[:-1]))

    # 1. Approximate source mesh with decomposed convex hulls
    style_decomp_path = DACH(mesh_path=style_path)
    # 2. Union of source convex hulls
    # src_union_path = MeshUnion(mesh_path=src_decomp_path)
    style_union_path = MeshUnion_marching_cube(mesh_path=style_decomp_path, resolution=256, do_dilation=True)
    # 6. Register source hull mesh to style hull mesh
    reg_style_path = Register(src_path=template_path, trg_path=style_union_path, save2trg=True)
    # 7. Deform source tetrahedron with source hull mesh controling points
    msh_arap_style_path, msh_arap_style_path_obj = ARAP_deform(msh_path=template_msh_path, src_template_path=template_path, trg_template_path=reg_style_path, tmp_trg_scale=2., sv_path='/'.join(reg_style_path.split('/')[:-1]))

    # 8. Align arap deformed source msh to original source msh
    msh_arap_style_path_align = Align(src_path=msh_arap_style_path, trg_path=msh_arap_src_path)
    Align(src_path=msh_arap_style_path_obj, trg_path=msh_arap_src_path_obj)
    print('Stylized MSH: ' + msh_arap_style_path_align, '\nContent MSH: ' + msh_arap_src_path)


def ToDACHTet(mesh_path):
    expand_path = Expand(mesh_path, scale=2)
    decomp_path = DACH(mesh_path=expand_path, hull_num=4)
    union_path = MeshUnion(mesh_path=decomp_path)
    simp_path = MeshSimplify(union_path, preserve_feature=False)
    msh_path = Convert2Tetrahedron(mesh_path=simp_path)
    print('Convert to DACH Tetrahedron. Saved in: ' + msh_path)
    return msh_path


if __name__ == '__main__':
    src_path = './logs/style_nerf_uglydog/meshes/ngp_1.obj'
    # ToDACHTet(mesh_path=src_path)
    style_path = './logs/style_nerf_cocacola/meshes/ngp_1.obj'
    InDirect_deform(src_path=src_path, style_path=style_path)
