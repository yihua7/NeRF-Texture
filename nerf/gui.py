import math
import torch
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from nerf.utils import *


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

        # wrong: rotate along global x/y axis
        #self.rot = R.from_euler('xy', [-dy * 0.1, -dx * 0.1], degrees=True) * self.rot
    
    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.001 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])

        # wrong: pan in global coordinate system
        #self.center += 0.001 * np.array([-dx, -dy, dz])
    


class NeRFGUI:
    def __init__(self, opt, trainer, gui_mode=True, debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.trainer = trainer
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32).cuda() # default white bg
        self.training = False
        self.step = 0 # training step 

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel

        self.dynamic_resolution = False
        self.downscale = 1
        self.train_steps = 16
        self.simple_render = False
        self.gui_mode = gui_mode

        dpg.create_context()
        self.register_dpg()
        self.test_step()
        

    def __del__(self):
        dpg.destroy_context()


    def train_step(self):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        outputs = self.trainer.train_gui(self.trainer.train_loader, step=self.train_steps)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.step += self.train_steps
        self.need_update = True

        dpg.set_value("_log_train_time", f'{t:.4f}ms')
        dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps

    
    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?

        if self.need_update or self.spp < self.opt.max_spp:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            if self.gui_mode:
                outputs = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color, self.spp, self.downscale, self.simple_render)
            else:
                outputs = {'image': self.trainer.train_loader._data.nn_image(self.cam.pose, self.W, self.H)}

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = outputs['image']
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + outputs['image']) / (self.spp + 1)
                self.spp += 1

            dpg.set_value("_log_infer_time", f'{t:.4f}ms')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value("_log_spp", self.spp)
            dpg.set_value("_texture", self.render_buffer)
    
    def take_photo(self):
        print('Taking photo ...')
        sv_path = self.trainer.take_photo(self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color)
        print('Saved views in ' + sv_path)
        return sv_path
    
    def render_train(self):
        print('Render train ...')

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        sv_path = self.trainer.render_train(self.bg_color)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)
        print('Saved views in ' + sv_path + ' Time cost: ' + str(t))
        return sv_path
           
    
    def render_round(self, render_light=False, fix_phi=False, fix_theta=False):
        print('Render round ...')

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        sv_path = self.trainer.render_round(self.cam.intrinsics, 1920, 1080, self.bg_color, render_light=render_light, fix_phi=fix_phi, fix_theta=fix_theta)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)
        print('Saved views in ' + sv_path + ' Time cost: ' + str(t))
        return sv_path

        
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=600, height=500):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            if not self.opt.test:
                with dpg.group(horizontal=True):
                    dpg.add_text("Train time: ")
                    dpg.add_text("no data", tag="_log_train_time")                    

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            with dpg.group(horizontal=True):
                dpg.add_text("SPP: ")
                dpg.add_text("1", tag="_log_spp")

            # train button
            if not self.opt.test:
                with dpg.collapsing_header(label="Train", default_open=True):

                    # train / stop
                    with dpg.group(horizontal=True):
                        dpg.add_text("Train: ")

                        def callback_train(sender, app_data):
                            if self.training:
                                self.training = False
                                dpg.configure_item("_button_train", label="start")
                            else:
                                self.training = True
                                dpg.configure_item("_button_train", label="stop")
                        
                        def callback_gui_mode(sender, app_data):
                            self.gui_mode = not self.gui_mode
                            self.need_update = True
                        
                        def callback_simple_render(sender, app_data):
                            self.simple_render = not self.simple_render
                            self.need_update = True
                        
                        def callback_distillation(sender, app_data):
                            if self.trainer.distillation:
                                self.trainer.distillation = False
                                dpg.configure_item("_button_distillation", label="distill")
                            else:
                                self.trainer.distillation = True
                                self.trainer.load_teacher_model()
                                dpg.configure_item("_button_distillation", label="no distill")

                        dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                        dpg.bind_item_theme("_button_train", theme_button)

                        dpg.add_button(label="gui_mode", tag="_button_gui_mode", callback=callback_gui_mode)
                        dpg.bind_item_theme("_button_gui_mode", theme_button)

                        dpg.add_button(label="simple_render", tag="_button_simple_render", callback=callback_simple_render)
                        dpg.bind_item_theme("_button_simple_render", theme_button)

                        dpg.add_button(label="distill", tag="_button_distillation", callback=callback_distillation)
                        dpg.bind_item_theme("_button_distillation", theme_button)

                        def callback_reset(sender, app_data):
                            @torch.no_grad()
                            def weight_reset(m: nn.Module):
                                reset_parameters = getattr(m, "reset_parameters", None)
                                if callable(reset_parameters):
                                    m.reset_parameters()
                            self.trainer.model.apply(fn=weight_reset)
                            self.trainer.model.reset_extra_state() # for cuda_ray density_grid and step_counter
                            self.need_update = True

                        dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)
                        dpg.bind_item_theme("_button_reset", theme_button)

                    with dpg.group(horizontal=True):
                        dpg.add_text("", tag="_log_train_log")

                    # save ckpt
                    with dpg.group(horizontal=True):
                        dpg.add_text("Checkpoint: ")

                        def callback_save(sender, app_data):
                            self.trainer.save_checkpoint(full=True, best=False)
                            dpg.set_value("_log_ckpt", "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
                            self.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="save", tag="_button_save", callback=callback_save)
                        dpg.bind_item_theme("_button_save", theme_button)

                        dpg.add_text("", tag="_log_ckpt")
                    
                    # save mesh
                    with dpg.group(horizontal=True):
                        dpg.add_text("Actions: ")

                        def callback_mesh(sender, app_data):
                            sv_path = self.trainer.save_mesh(resolution=256, threshold=10)
                            dpg.set_value("_log_mesh", "saved " + sv_path)
                            self.trainer.epoch += 1 # use epoch to indicate different calls.

                        def callback_pcl(sender, app_data):
                            sv_path = self.trainer.save_point_cloud()
                            dpg.set_value("_log_mesh", "saved " + sv_path)
                            self.trainer.epoch += 1 # use epoch to indicate different calls.
                        
                        def callback_poses(sender, app_data):
                            sv_path = self.trainer.save_poses()
                            dpg.set_value("_log_mesh", "saved " + sv_path)

                        dpg.add_button(label="save_mesh", tag="_button_mesh", callback=callback_mesh)
                        dpg.bind_item_theme("_button_mesh", theme_button)

                        dpg.add_button(label="save_pcl", tag="_button_pcl", callback=callback_pcl)
                        dpg.bind_item_theme("_button_pcl", theme_button)

                        dpg.add_button(label="save_poses", tag="_button_poses", callback=callback_poses)
                        dpg.bind_item_theme("_button_poses", theme_button)

                        dpg.add_text("", tag="_log_mesh")

                    with dpg.group(horizontal=True):
                        dpg.add_text("Render scripts: ")
                        
                        def callback_render_train(sender, app_data):
                            sv_path = self.render_train()
                            dpg.set_value("_log_mesh", "saved " + sv_path)
                        dpg.add_button(label="render_train", tag="_button_render_train", callback=callback_render_train)
                        dpg.bind_item_theme("_button_render_train", theme_button)
                        
                        def callback_take_photo(sender, app_data):
                            sv_path = self.take_photo()
                            dpg.set_value("_log_mesh", "saved " + sv_path)
                        dpg.add_button(label="take_photo", tag="_button_take_photo", callback=callback_take_photo)
                        dpg.bind_item_theme("_button_take_photo", theme_button)

                    with dpg.group(horizontal=True):

                        def callback_render_round(sender, app_data):
                            sv_path = self.render_round()
                            dpg.set_value("_log_mesh", "saved " + sv_path)
                        def render_round_fixphi(sender, app_data):
                            sv_path = self.render_round(fix_phi=True)
                            dpg.set_value("_log_mesh", "saved " + sv_path)
                        def render_round_theta(sender, app_data):
                            sv_path = self.render_round(fix_theta=True)
                            dpg.set_value("_log_mesh", "saved " + sv_path)
                        dpg.add_button(label="render_r", tag="_button_render_round", callback=callback_render_round)
                        dpg.bind_item_theme("_button_render_round", theme_button)
                        dpg.add_button(label="render_rphi", tag="_button_render_round_phi", callback=render_round_fixphi)
                        dpg.bind_item_theme("_button_render_round_phi", theme_button)
                        dpg.add_button(label="render_rtheta", tag="_button_render_round_theta", callback=render_round_theta)
                        dpg.bind_item_theme("_button_render_round_theta", theme_button)

                    with dpg.group(horizontal=True):
                        dpg.add_text("Field: ")

                        def callback_sv_field(sender, app_data):
                            sv_path = self.trainer.save_field()
                            dpg.set_value("_log_field", "saved " + sv_path)

                        def callback_ld_field(sender, app_data):
                            print('Loading synthesized field...')
                            ld_path = self.trainer.load_field()
                            dpg.set_value("_log_field", "loaded from " + ld_path)
                            self.need_update = True

                        def callback_ld_patch(sender, app_data):
                            ld_path = self.trainer.load_patch()
                            dpg.set_value("_log_field", "loaded from " + ld_path)
                            self.need_update = True

                        def callback_ld_shape(sender, app_data):
                            ld_path = self.trainer.load_shape()
                            dpg.set_value("_log_field", "loaded from " + ld_path)
                            self.need_update = True
                        
                        dpg.add_button(label="sample patches", tag="_button_field_sv", callback=callback_sv_field)
                        dpg.bind_item_theme("_button_field_sv", theme_button)
                        dpg.add_button(label="load synthesis", tag="_button_field_ld", callback=callback_ld_field)
                        dpg.bind_item_theme("_button_field_ld", theme_button)
                        dpg.add_button(label="load_patch", tag="_button_patch_ld", callback=callback_ld_patch)
                        dpg.bind_item_theme("_button_patch_ld", theme_button)
                        dpg.add_button(label="load_shape", tag="_button_shape_ld", callback=callback_ld_shape)
                        dpg.bind_item_theme("_button_shape_ld", theme_button)
                    
                    with dpg.group(horizontal=True):
                        
                        def callback_unhash(sender, app_data):
                            if hasattr(self.trainer.model, 'meshfea_field') and hasattr(self.trainer.model.meshfea_field, 'unhash'):
                                self.trainer.model.meshfea_field.unhash()
                                self.trainer.model.initialize_states(fp16=True)
                                dpg.set_value("_log_field", f"Unhash")
                                self.need_update = True
                            else:
                                dpg.set_value("_log_field", "Unhash not supported")
                        def callback_import_unhash(sender, app_data):
                            ld_path = self.trainer.load_unhash()
                            dpg.set_value("_log_field", "loaded from " + ld_path)
                            self.need_update = True
                        def callback_switch_shape_feature(sender, app_data):
                            self.trainer.model.switch_shape_feature()
                            dpg.set_value("_log_field", f"Switch to {self.trainer.model.meshfea_field.imported_type}")
                            self.need_update = True
                        def callback_switch_import(sender, app_data):
                            self.trainer.model.switch_import()
                            dpg.set_value("_log_field", f"Switch imported to {self.trainer.model.meshfea_field.imported}")
                            self.need_update = True
                        dpg.add_button(label="unhash", tag="_button_unhash", callback=callback_unhash)
                        dpg.bind_item_theme("_button_unhash", theme_button)
                        dpg.add_button(label="import_unhash", tag="_button_import_unhash", callback=callback_import_unhash)
                        dpg.bind_item_theme("_button_import_unhash", theme_button)
                        dpg.add_button(label="switch_shape_fea", tag="_button_switch_shape_fea", callback=callback_switch_shape_feature)
                        dpg.bind_item_theme("_button_switch_shape_fea", theme_button)
                        dpg.add_button(label="switch_import", tag="_button_switch", callback=callback_switch_import)
                        dpg.bind_item_theme("_button_switch", theme_button)
                        dpg.add_text("", tag="_log_field")
                    
                    with dpg.group(horizontal=False):
                        def callback_euler_x(sender):
                            self.trainer.euler[0] = dpg.get_value(sender)
                            self.need_update = True
                        dpg.add_slider_float(
                            label="Euler X",
                            default_value=0.,
                            max_value=-np.pi,
                            min_value=np.pi,
                            callback=callback_euler_x,
                        )
                        def callback_euler_y(sender):
                            self.trainer.euler[1] = dpg.get_value(sender)
                            self.need_update = True
                        dpg.add_slider_float(
                            label="Euler Y",
                            default_value=0.,
                            max_value=-np.pi,
                            min_value=np.pi,
                            callback=callback_euler_y,
                        )
                        def callback_euler_z(sender):
                            self.trainer.euler[2] = dpg.get_value(sender)
                            self.need_update = True
                        dpg.add_slider_float(
                            label="Euler Z",
                            default_value=0.,
                            max_value=-np.pi,
                            min_value=np.pi,
                            callback=callback_euler_z,
                        )

                        with dpg.group(horizontal=True):
                            def callback_uv_utilize_rate(sender):
                                self.trainer.set_uv_utilize_rate(dpg.get_value('uv_rate'))
                                self.need_update = True
                                print('uv_rate set')
                            dpg.add_input_float(label='', tag='uv_rate', width=100, default_value=1.)
                            dpg.add_button(label='uv_rate', tag='_button_set_uv_rate', callback=callback_uv_utilize_rate)
                            dpg.bind_item_theme("_button_set_uv_rate", theme_button)

                            def callback_k_for_uv(sender):
                                k_for_uv = self.trainer.set_k_for_uv(dpg.get_value('k_for_uv'))
                                self.need_update = True
                                print(f'k_for_uv set {k_for_uv}')
                            dpg.add_input_int(label='', tag='k_for_uv', width=100, default_value=5)
                            dpg.add_button(label='k_for_uv', tag='_button_set_k_for_uv', callback=callback_k_for_uv)
                            dpg.bind_item_theme("_button_set_k_for_uv", theme_button)

                            def callback_sdf_factor(sender):
                                self.trainer.set_sdf_factor(dpg.get_value('sdf_factor'))
                                self.need_update = True
                                print('SDF factor set')
                            dpg.add_input_float(label='', tag='sdf_factor', width=100, default_value=1.)
                            dpg.add_button(label='sdf factor', tag='_button_set_sdf_factor', callback=callback_sdf_factor)
                            dpg.bind_item_theme("_button_set_sdf_factor", theme_button)

                        with dpg.group(horizontal=True):
                            def callback_sdf_offset(sender):
                                self.trainer.set_sdf_offset(dpg.get_value('sdf_offset'))
                                self.need_update = True
                                print('SDF offset set')
                            dpg.add_input_float(label='', tag='sdf_offset', width=100, default_value=0.)
                            dpg.add_button(label='sdf offset', tag='_button_set_sdf_offset', callback=callback_sdf_offset)
                            dpg.bind_item_theme("_button_set_sdf_offset", theme_button)

                            def callback_h_threshold(sender):
                                self.trainer.set_h_threshold(dpg.get_value('h_threshold'))
                                self.need_update = True
                                print('h threshold set')
                            dpg.add_input_float(label='', tag='h_threshold', width=100, default_value=0.03)
                            dpg.add_button(label='set h_threshold', tag='_button_set_h_threshold', callback=callback_h_threshold)
                            dpg.bind_item_theme("_button_set_h_threshold", theme_button)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("Visual: ")

                        # def callback_switch_visual_mode(sender, app_data):
                        #     visual_mode = self.trainer.model.switch_visual_mode()
                        #     dpg.set_value("_visual_mode", visual_mode)
                        #     self.need_update = True
                        # dpg.add_button(label="switch_visual", tag="_button_switch_visual", callback=callback_switch_visual_mode)
                        # dpg.bind_item_theme("_button_switch_visual", theme_button)

                        def create_callback_visual_mode(i):
                            def callback_visual_mode(sender, app_data):
                                visual_mode = self.trainer.model.switch_visual_mode(i)
                                dpg.set_value("_visual_mode", visual_mode)
                                self.need_update = True
                            return callback_visual_mode
                        for i in range(len(self.trainer.model.visual_modes)):
                            dpg.add_button(label=self.trainer.model.visual_modes[i], tag=f"_button_switch_visual_{i}", callback=create_callback_visual_mode(i))
                            dpg.bind_item_theme(f"_button_switch_visual_{i}", theme_button)

                        def callback_visualize_features(sender, app_data):
                            sv_path = self.trainer.visualize_features()
                            dpg.set_value("_visual_mode", sv_path)

                        dpg.add_button(label="vis_fea", tag="_button_visual_features", callback=callback_visualize_features)
                        dpg.bind_item_theme("_button_visual_features", theme_button)

                        dpg.add_text("", tag="_visual_mode")

                    with dpg.group(horizontal=True):
                        dpg.add_text("Light Model: ")
                        def callback_switch_light_model(sender, app_data):
                            if self.trainer.model.light_model == 'None':
                                dpg.set_value("_light_model", f'light model is None. invalid operation!')
                            else:
                                self.trainer.model.render_light_model = not self.trainer.model.render_light_model
                                dpg.set_value("_light_model", f'set to {self.trainer.model.render_light_model}')
                                self.need_update = True

                        dpg.add_button(label="switch_light_model", tag="_button_switch_light_model", callback=callback_switch_light_model)
                        dpg.bind_item_theme("_button_switch_light_model", theme_button)

                        def create_callback_light_visual_mode(i):
                            def callback_light_visual_mode(sender, app_data):
                                light_visual_mode = self.trainer.model.switch_light_mode(i)
                                dpg.set_value("_light_model", light_visual_mode)
                                self.need_update = True
                            return callback_light_visual_mode
                        for i in range(len(self.trainer.model.light_visual_modes)):
                            dpg.add_button(label=self.trainer.model.light_visual_modes[i], tag=f"_button_switch_light_visual_{i}", callback=create_callback_light_visual_mode(i))
                            dpg.bind_item_theme(f"_button_switch_light_visual_{i}", theme_button)

                    with dpg.group(horizontal=True):
                        def callback_save_envmap(sender, app_data):
                            if self.trainer.model.light_model == 'None':
                                dpg.set_value("_light_model", f'light model is None. invalid operation!')
                            else:
                                sv_path = self.trainer.save_envmap()
                                dpg.set_value("_light_model", f'save envmap to {sv_path}')
                                self.need_update = True
                        dpg.add_button(label="save_envmap", tag="_button_save_envmap", callback=callback_save_envmap)
                        dpg.bind_item_theme("_button_save_envmap", theme_button)

                        def callback_load_envmap(sender, app_data):
                            if self.trainer.model.light_model == 'None':
                                dpg.set_value("_light_model", f'light model is None. invalid operation!')
                            else:
                                ld_path = self.trainer.load_envmap()
                                dpg.set_value("_light_model", f'load envmap from {ld_path}')
                                self.need_update = True
                        dpg.add_button(label="load_envmap", tag="_button_load_envmap", callback=callback_load_envmap)
                        dpg.bind_item_theme("_button_load_envmap", theme_button)

                        def callback_switch_envmap(sender, app_data):
                            if self.trainer.model.light_model == 'None':
                                dpg.set_value("_light_model", f'light model is None. invalid operation!')
                            else:
                                envmap_import = self.trainer.model.light_net.switch_envmap_import()
                                dpg.set_value("_light_model", f'switch envmap import to {envmap_import}')
                                self.need_update = True
                        dpg.add_button(label="switch_envmap", tag="_button_switch_envmap", callback=callback_switch_envmap)
                        dpg.bind_item_theme("_button_switch_envmap", theme_button)

                        def callback_switch_shade_visibility(sender, app_data):
                            if self.trainer.model.light_model == 'None':
                                dpg.set_value("_light_model", f'light model is None. invalid operation!')
                            else:
                                self.trainer.model.shade_visibility = not self.trainer.model.shade_visibility
                                dpg.set_value("_light_model", f'coarse decay: {self.trainer.model.shade_visibility}')
                                self.need_update = True
                        dpg.add_button(label="shade_visibility", tag="_button_switch_shade_visibility", callback=callback_switch_shade_visibility)
                        dpg.bind_item_theme("_button_switch_shade_visibility", theme_button)

                        def callback_switch_use_coarse_normal(sender, app_data):
                            if self.trainer.model.light_model == 'None':
                                dpg.set_value("_light_model", f'light model is None. invalid operation!')
                            else:
                                self.trainer.model.use_coarse_normal = not self.trainer.model.use_coarse_normal
                                self.trainer.model.use_grad_normal = False
                                dpg.set_value("_light_model", f'use_coarse_normal: {self.trainer.model.use_coarse_normal}')
                                self.need_update = True
                        dpg.add_button(label="use_coarse", tag="_button_switch_use_coarse_normal", callback=callback_switch_use_coarse_normal)
                        dpg.bind_item_theme("_button_switch_use_coarse_normal", theme_button)

                        def callback_switch_use_grad_normal(sender, app_data):
                            if self.trainer.model.light_model == 'None':
                                dpg.set_value("_light_model", f'light model is None. invalid operation!')
                            else:
                                self.trainer.model.use_grad_normal = not self.trainer.model.use_grad_normal
                                self.trainer.model.use_coarse_normal = False
                                dpg.set_value("_light_model", f'use_grad_normal: {self.trainer.model.use_grad_normal}')
                                self.need_update = True
                        dpg.add_button(label="use_grad", tag="_button_switch_use_grad_normal", callback=callback_switch_use_grad_normal)
                        dpg.bind_item_theme("_button_switch_use_grad_normal", theme_button)

                    dpg.add_text("", tag="_light_model")
                    
                    def callback_fc_normal_weight(sender):
                        self.trainer.model.fc_weight = dpg.get_value(sender)
                        self.need_update = True
                    dpg.add_slider_float(
                        label="FC weight",
                        default_value=1.,
                        max_value=0.,
                        min_value=1.,
                        callback=callback_fc_normal_weight,
                    )

            
            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution, callback=callback_set_dynamic_resolution)
                    dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32).cuda() # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor", no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

                # dt_gamma slider
                def callback_set_dt_gamma(sender, app_data):
                    self.opt.dt_gamma = app_data
                    self.need_update = True

                dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f", default_value=self.opt.dt_gamma, callback=callback_set_dt_gamma)

                # aabb slider
                def callback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    self.trainer.model.aabb_infer[user_data] = app_data

                    # also change train aabb ? [better not...]
                    #self.trainer.model.aabb_train[user_data] = app_data

                    self.need_update = True

                dpg.add_separator()
                dpg.add_text("Axis-aligned bounding box:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="x", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=0)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=3)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="y", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=1)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=4)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="z", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=2)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=5)
                

            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")


        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

        
        dpg.create_viewport(title='nerf-texture-synthesis', width=self.W, height=self.H, resizable=False)
        
        # TODO: seems dearpygui doesn't support resizing texture...
        # def callback_resize(sender, app_data):
        #     self.W = app_data[0]
        #     self.H = app_data[1]
        #     # how to reload texture ???

        # dpg.set_viewport_resize_callback(callback_resize)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):

        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
