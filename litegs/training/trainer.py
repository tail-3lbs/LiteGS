import torch
from torch.utils.data import DataLoader
import fused_ssim
from torchmetrics.image import psnr
from tqdm import tqdm
import numpy as np
import math
import os
import torch.cuda.nvtx as nvtx

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from .. import arguments
from .. import data
from .. import io_manager
from .. import scene
from . import optimizer
from ..data import CameraFrameDataset
from .. import render
from ..utils.statistic_helper import StatisticsHelperInst
from . import densify


def pretty_scientific(
        nums: list[float],
    precision: int = 4,
    sep: str = " "
) -> str:
    """
    Return a single-line string of numbers in scientific notation.

    Args:
        nums:       List of float numbers to format.
        precision:  Number of digits after the decimal in the exponent format.
        sep:        Separator between formatted numbers.

    Returns:
        A string like "1.2345e+00 2.3456e-01 3.4567e+02"
    """
    # Build a dynamic format string, e.g. "{:.4e}"
    fmt = f"{{:.{precision}e}}"
    # Apply to each number and join without newlines
    return "(" + sep.join(fmt.format(x) for x in nums) + ")"


def __l1_loss(network_output:torch.Tensor, gt:torch.Tensor)->torch.Tensor:
    return torch.abs((network_output - gt)).mean()

def start(lp:arguments.ModelParams,op:arguments.OptimizationParams,pp:arguments.PipelineParams,dp:arguments.DensifyParams,
          test_epochs=[],save_ply=[],save_checkpoint=[],start_checkpoint=None):

    # Setup TensorBoard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(os.path.join(lp.model_path, "tensorboard"))
        print("TensorBoard logging enabled at:", os.path.join(lp.model_path, "tensorboard"))
    else:
        print("TensorBoard not available: not logging progress")

    
    cameras_info:dict[int,data.CameraInfo]=None
    camera_frames:list[data.CameraFrame]=None
    cameras_info,camera_frames,init_xyz,init_color=io_manager.load_colmap_result(lp.source_path,lp.images)#lp.sh_degree,lp.resolution

    #preload
    for camera_frame in camera_frames:
        camera_frame.load_image(lp.resolution)

    #Dataset
    if lp.eval:
        training_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 != 0]
        test_frames=[c for idx, c in enumerate(camera_frames) if idx % 8 == 0]
    else:
        training_frames=camera_frames
        test_frames=None
    trainingset=CameraFrameDataset(cameras_info,training_frames,lp.resolution,pp.device_preload)
    train_loader = DataLoader(trainingset, batch_size=1,shuffle=True,pin_memory=not pp.device_preload)
    test_loader=None
    if lp.eval:
        testset=CameraFrameDataset(cameras_info,test_frames,lp.resolution,pp.device_preload)
        test_loader = DataLoader(testset, batch_size=1,shuffle=False,pin_memory=not pp.device_preload)
    norm_trans,norm_radius=trainingset.get_norm()

    #torch parameter
    cluster_origin=None
    cluster_extend=None
    init_points_num=init_xyz.shape[0]
    if start_checkpoint is None:
        init_xyz=torch.tensor(init_xyz,dtype=torch.float32,device='cuda')
        init_color=torch.tensor(init_color,dtype=torch.float32,device='cuda')
        xyz,scale,rot,sh_0,sh_rest,opacity=scene.create_gaussians(init_xyz,init_color,lp.sh_degree)
        if pp.cluster_size:
            xyz,scale,rot,sh_0,sh_rest,opacity=scene.cluster.cluster_points(pp.cluster_size,xyz,scale,rot,sh_0,sh_rest,opacity)
        xyz=torch.nn.Parameter(xyz)
        scale=torch.nn.Parameter(scale)
        rot=torch.nn.Parameter(rot)
        sh_0=torch.nn.Parameter(sh_0)
        sh_rest=torch.nn.Parameter(sh_rest)
        opacity=torch.nn.Parameter(opacity)
        opt,schedular=optimizer.get_optimizer(xyz,scale,rot,sh_0,sh_rest,opacity,norm_radius,op,pp)
        start_epoch=0
    else:
        xyz,scale,rot,sh_0,sh_rest,opacity,start_epoch,opt,schedular=io_manager.load_checkpoint(start_checkpoint)
        if pp.cluster_size:
            cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
    actived_sh_degree=0

    #init
    total_epoch=int(op.iterations/len(trainingset))
    print(f"total_epoch=iterations/len(trainingset): {total_epoch}={op.iterations}/{len(trainingset)}")
    if dp.densify_until<0:
        dp.densify_until=int(total_epoch*0.8/dp.opacity_reset_interval)*dp.opacity_reset_interval+1
    density_controller=densify.DensityControllerTamingGS(norm_radius,dp,pp.cluster_size>0,init_points_num)
    StatisticsHelperInst.reset(xyz.shape[-2],xyz.shape[-1],density_controller.is_densify_actived)
    progress_bar = tqdm(range(start_epoch, total_epoch), desc="Training progress")
    progress_bar.update(0)
    
    # Timing variables for per-iteration monitoring
    timing_interval = 10
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for epoch in range(start_epoch,total_epoch):
        print("="*90)
        # Epoch starts from 0 and iteration starts from 0 too.
        total_gaussians = xyz.shape[1] * xyz.shape[2]
        print(f"[EPOCH {epoch}] [ITER {schedular.last_epoch}] Training started with gaussian number: {total_gaussians}")

        with torch.no_grad():
            if pp.cluster_size>0 and epoch%dp.densification_interval==0:
                scene.spatial_refine(pp.cluster_size>0,opt,xyz)
                cluster_origin,cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
            if actived_sh_degree<lp.sh_degree:
                actived_sh_degree=min(int(epoch/5),lp.sh_degree)

        with StatisticsHelperInst.try_start(epoch):
            for view_matrix,proj_matrix,frustumplane,gt_image in train_loader:
                # Start timing for this iteration if we're at a timing interval
                if schedular.last_epoch % timing_interval == 0:
                    start_event.record()
                
                view_matrix=view_matrix.cuda()
                proj_matrix=proj_matrix.cuda()
                frustumplane=frustumplane.cuda()
                gt_image=gt_image.cuda()/255.0

                #cluster culling
                visible_chunkid,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(cluster_origin,cluster_extend,frustumplane,
                                                                                                               xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                img,transmitance,depth,normal,primitive_visible=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                            actived_sh_degree,gt_image.shape[2:],pp)
                
                l1_loss=__l1_loss(img,gt_image)
                ssim_loss:torch.Tensor=1-fused_ssim.fused_ssim(img,gt_image)
                loss=(1.0-op.lambda_dssim)*l1_loss+op.lambda_dssim*ssim_loss
                loss+=(culled_scale).square().mean()*op.reg_weight
                loss.backward()
                # with torch.no_grad():
                #     if pp.cluster_size>0:
                #         print(count,opacity.sigmoid()[...,5,112])
                #     else:
                #         print(count,opacity.sigmoid()[...,752])
                if StatisticsHelperInst.bStart:
                    StatisticsHelperInst.backward_callback()
                if pp.sparse_grad:
                    opt.step(visible_chunkid,primitive_visible)
                else:
                    opt.step()

                if tb_writer:
                    tb_writer.add_scalar('counts/4-total_gaussians', total_gaussians, schedular.last_epoch)

                opt.zero_grad(set_to_none = True)

                # Before step(), the last_epoch is 0. After step(), the last_epoch is 1.
                schedular.step()
                
                # Record timing if this was a timing iteration
                if (schedular.last_epoch - 1) % timing_interval == 0:
                    end_event.record()
                    torch.cuda.synchronize()
                    per_iter_time_ms = start_event.elapsed_time(end_event)
                    per_iter_time_us = per_iter_time_ms * 1000
                    if tb_writer:
                        tb_writer.add_scalar('timing/per_iter_time_us', per_iter_time_us, schedular.last_epoch-1)

            # If last_epoch is 10, that means we have run through 10 iterations: 0-9.
            print(f"[EPOCH {epoch}] Training done. Total iterations used till now: {schedular.last_epoch}")

        if epoch in test_epochs:
            print("\n\n")
            with torch.no_grad():
                _cluster_origin=None
                _cluster_extend=None
                if pp.cluster_size:
                    _cluster_origin,_cluster_extend=scene.cluster.get_cluster_AABB(xyz,scale.exp(),torch.nn.functional.normalize(rot,dim=0))
                psnr_metrics=psnr.PeakSignalNoiseRatio(data_range=(0.0,1.0)).cuda()
                loaders={"Trainingset":train_loader}
                if lp.eval:
                    loaders["Testset"]=test_loader
                for name,loader in loaders.items():
                    psnr_list=[]
                    for view_matrix,proj_matrix,frustumplane,gt_image in loader:
                        view_matrix=view_matrix.cuda()
                        proj_matrix=proj_matrix.cuda()
                        frustumplane=frustumplane.cuda()
                        gt_image=gt_image.cuda()/255.0
                        _,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity=render.render_preprocess(_cluster_origin,_cluster_extend,frustumplane,
                                                                                                                xyz,scale,rot,sh_0,sh_rest,opacity,op,pp)
                        img,transmitance,depth,normal,_=render.render(view_matrix,proj_matrix,culled_xyz,culled_scale,culled_rot,culled_sh_0,culled_sh_rest,culled_opacity,
                                                                    actived_sh_degree,gt_image.shape[2:],pp)
                        psnr_list.append(psnr_metrics(img,gt_image).unsqueeze(0))
                    psnr_mean = torch.concat(psnr_list,dim=0).mean()
                    print("[EPOCH {}] {} Evaluating: PSNR {}".format(epoch,name,psnr_mean))
            print("\n\n")

        xyz,scale,rot,sh_0,sh_rest,opacity=density_controller.step(opt,epoch)

        progress_bar.update()  

        if epoch in save_ply or epoch==total_epoch-1:
            if epoch==total_epoch-1:
                progress_bar.close()
                print("{} takes: {} s".format(lp.model_path,progress_bar.format_dict['elapsed']))
                ply_path=os.path.join(lp.model_path,"point_cloud","finish","point_cloud.ply")
            else:
                ply_path=os.path.join(lp.model_path,"point_cloud","iteration_{}".format(epoch),"point_cloud.ply")    

            if pp.cluster_size:
                tensors=scene.cluster.uncluster(xyz,scale,rot,sh_0,sh_rest,opacity)
            else:
                tensors=xyz,scale,rot,sh_0,sh_rest,opacity
            param_nyp=[]
            for tensor in tensors:
                param_nyp.append(tensor.detach().cpu().numpy())
            io_manager.save_ply(ply_path,*param_nyp)
            pass

        if epoch in save_checkpoint:
            io_manager.save_checkpoint(lp.model_path,epoch,opt,schedular)

        print(f"[EPOCH {epoch}] Epoch ended. Total iterations used till now: {schedular.last_epoch}")
        print("="*90)

    # Save final Gaussian count to file
    final_gaussian_count = xyz.shape[1] * xyz.shape[2]
    gaussian_count_file = os.path.join(lp.model_path, "gaussian_count.txt")
    with open(gaussian_count_file, 'w') as f:
        f.write(f"Final Gaussian count: {final_gaussian_count}\n")
        f.write(f"Total epochs: {total_epoch}\n")
        f.write(f"Final iteration: {schedular.last_epoch}\n")
    print(f"Saved Gaussian count ({final_gaussian_count}) to {gaussian_count_file}")

    # Close TensorBoard writer
    if tb_writer:
        tb_writer.close()
        print("TensorBoard writer closed")

    return