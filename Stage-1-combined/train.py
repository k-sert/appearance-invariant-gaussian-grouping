#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, loss_cls_3d
from gaussian_renderer import render, network_gui
from os import makedirs
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams,args_init
import logging,time,shutil,torchvision
import matplotlib.pyplot as plt
from render import *
from metrics import evaluate as evaluate_metrics
from metrics_half import evaluate as evaluate_metrics_half
import pickle

import lpips
from arguments import *
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def _segmentation_palette(n):
    """Return a (n, 3) uint8 color palette, one colour per class."""
    rng = np.random.default_rng(42)
    palette = rng.integers(50, 230, size=(max(n, 1), 3), dtype=np.uint8)
    palette[0] = [0, 0, 0]  # class 0 = black background
    return palette


def _colorize_segmentation(logits, palette):
    """logits: (1, C, H, W) -> RGB tensor (3, H, W) in [0, 1]."""
    ids = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (H, W)
    rgb = palette[ids]                                    # (H, W, 3)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def render_segmentation_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, palette):
    """Render and save colourised segmentation maps alongside normal renders."""
    segmentation_path = os.path.join(model_path, name, "ours_{}".format(iteration), "segmentation")
    makedirs(segmentation_path, exist_ok=True)
    classifier.eval()
    with torch.no_grad():
        for idx, view in enumerate(tqdm(views, desc="Rendering segmentation")):
            pkg = render(view, gaussians, pipeline, background)
            obj = pkg["render_object"]
            if obj is None:
                continue
            logits = classifier(obj.unsqueeze(0))
            segmentation_image = _colorize_segmentation(logits, palette)
            torchvision.utils.save_image(segmentation_image, os.path.join(segmentation_path, '{0:05d}'.format(idx) + ".png"))
    classifier.train()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, debug_from,args):
    '''
    dataset:Groupparams  ModelParams
    opt:Groupparams  OptimizationParams
    pipe:Groupparams  PipelineParams

    '''
    #print(f"{args}")
    saving_iterations+=[opt.iterations]
    dataset_name=args.scene_name
    dataset.data_perturb=args.data_perturb
    dataset.object_path=getattr(args,'object_path','')
    log_file_path=os.path.join(dataset.model_path,"logs",f"({time.strftime('%Y-%m-%d_%H-%M-%S')})_Iteration({opt.iterations})_({dataset_name}).log")
    os.makedirs(os.path.dirname(log_file_path),exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Experiment Configuration: {args}")
    logging.info(f"Model initialization and Data reading....")
    #save args
    with open(os.path.join(args.model_path,'cfg_arg.pkl'), 'wb') as file:
        pickle.dump(args, file)
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)        
    gaussians = GaussianModel(dataset.sh_degree,args)          
    scene = Scene(dataset, gaussians,shuffle=False)                     
    gaussians.training_setup(opt)

    # Segmentation classifier (gaussian-grouping style)
    # Auto-detect num_classes from the loaded segmentation masks
    classifier = None
    cls_criterion = None
    cls_optimizer = None
    if getattr(args, 'object_path', ''):
        all_ids = set()
        for cam in scene.getTrainCameras():
            if cam.objects is not None:
                all_ids.update(cam.objects.unique().tolist())
        num_classes = int(max(all_ids)) + 1 if all_ids else 0
        if num_classes > 0:
            print(f"Segmentation: detected {num_classes} classes from masks.")
            classifier = torch.nn.Conv2d(gaussians.object_embed_dim, num_classes, kernel_size=1).cuda()
            cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
            cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-4)
            segmentation_palette = _segmentation_palette(num_classes)

    render_temp_path=os.path.join(dataset.model_path,"train_temp_rendering")
    gt_temp_path=os.path.join(dataset.model_path,"train_temp_gt")
    segmentation_temp_path=os.path.join(dataset.model_path,"train_temp_segmentation")
    if os.path.exists(render_temp_path):
        shutil.rmtree(render_temp_path)
    if os.path.exists(gt_temp_path):
        shutil.rmtree(gt_temp_path)
    if os.path.exists(segmentation_temp_path):
        shutil.rmtree(segmentation_temp_path)
    os.makedirs(render_temp_path,exist_ok=True)
    os.makedirs(gt_temp_path,exist_ok=True)
    os.makedirs(segmentation_temp_path,exist_ok=True)
    
    if args.use_features_mask:
        render_temp_mask_path=os.path.join(dataset.model_path,"train_mask_temp_rendering")
        if os.path.exists(render_temp_mask_path):
            shutil.rmtree(render_temp_mask_path)
        os.makedirs(render_temp_mask_path,exist_ok=True)
        

    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_psnr_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    len_timedata=len(scene.getTrainCameras())
    record_loss=[]
    ech_loss=0
    if args.use_lpips_loss:#vgg alex
        lpips_criteria = lpips.LPIPS(net='vgg').to("cuda:0")
    logging.info(f"Start trainning....")

    if args.warm_up_iter>0:
        gaussians.set_learning_rate("box_coord",0.0)
        
    for iteration in range(first_iter, opt.iterations + 1):    
        #   
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration,args.warm_up_iter)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()          #
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))  #

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background


        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        if args.use_features_mask and iteration>args.features_mask_iters:#2500
            mask=gaussians.features_mask
            mask=torch.nn.functional.interpolate(mask,size=(image.shape[-2:]))
            Ll1 = l1_loss(image*mask, gt_image*mask)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image*mask, gt_image*mask))

        else:
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if args.use_features_mask and iteration>args.features_mask_iters:#2500
            loss+=(torch.square(1-gaussians.features_mask)).mean()*args.features_mask_loss_coef

        if args.use_scaling_loss :
            loss+=torch.abs(gaussians.get_scaling).mean()*args.scaling_loss_coef
        if args.use_lpips_loss:
            loss+=lpips_criteria(image,gt_image).mean()*args.lpips_loss_coef

        if ( gaussians.use_kmap_pjmap or gaussians.use_okmap) and args.use_box_coord_loss:
            loss+=torch.relu(torch.abs(gaussians.map_pts_norm)-1).mean()*args.box_coord_loss_coef

        # Segmentation loss (gaussian-grouping)
        loss_obj_3d = None
        if classifier is not None and render_pkg["render_object"] is not None and viewpoint_cam.objects is not None:
            gt_obj = viewpoint_cam.objects.cuda().long()
            objects = render_pkg["render_object"]  # (object_embed_dim, H, W)
            logits = classifier(objects.unsqueeze(0))  # (1, num_classes, H, W)
            loss_obj = cls_criterion(logits, gt_obj.unsqueeze(0)).mean()
            loss_obj = loss_obj / torch.log(torch.tensor(float(classifier.out_channels), device=loss_obj.device))
            loss = loss + loss_obj

            reg3d_interval = getattr(args, 'reg3d_interval', 2)
            if iteration % reg3d_interval == 0:
                obj_feats = gaussians.get_objects  # (N, object_embed_dim)
                logits3d = classifier(obj_feats.t().unsqueeze(0).unsqueeze(-1))  # (1, num_classes, N, 1)
                prob_obj3d = torch.softmax(logits3d.squeeze(0).squeeze(-1), dim=0).t()  # (N, num_classes)
                loss_obj_3d = loss_cls_3d(
                    gaussians.get_xyz.detach(), prob_obj3d,
                    k=getattr(args, 'reg3d_k', 5),
                    lambda_val=getattr(args, 'reg3d_lambda_val', 2.0),
                    max_points=getattr(args, 'reg3d_max_points', 300000),
                    sample_size=getattr(args, 'reg3d_sample_size', 1000),
                )
                loss += loss_obj_3d

        psnr_ = psnr(image,gt_image).mean().double()
        loss.backward()
 
        iter_end.record()
        ech_loss+=loss.item()
        if (iteration)%len_timedata==0:
            logging.info(f'Iteration {iteration}/{opt.iterations}, Loss: {ech_loss/len_timedata}')
            logging.info(f"Iteration {iteration}:Guassian points' number:{gaussians._xyz.shape[0]}")
            record_loss.append(ech_loss/len_timedata)
            ech_loss=0
            
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_for_log = 0.4 * psnr_ + 0.6 * ema_psnr_for_log
            total_point = gaussians._xyz.shape[0]
            if iteration%100==0 or iteration==1:
                torchvision.utils.save_image(image, os.path.join(render_temp_path, f"iter{iteration}_"+viewpoint_cam.image_name + ".png"))
                torchvision.utils.save_image(gt_image, os.path.join(gt_temp_path, f"iter{iteration}_"+viewpoint_cam.image_name + ".png"))
                if args.use_features_mask:
                    torchvision.utils.save_image(gaussians.features_mask.repeat(1,3,1,1), os.path.join(render_temp_mask_path, f"iter{iteration}_"+viewpoint_cam.image_name + ".png"))
                if classifier is not None and render_pkg["render_object"] is not None:
                    with torch.no_grad():
                        segmentation_image = _colorize_segmentation(classifier(render_pkg["render_object"].unsqueeze(0)), segmentation_palette)
                    torchvision.utils.save_image(segmentation_image, os.path.join(segmentation_temp_path, f"iter{iteration}_"+viewpoint_cam.image_name + ".png"))
            if iteration % 10 == 0:            
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}",
                                          "psnr": f"{psnr_:.{2}f}",
                                           "point":f"{total_point}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            gaussians.set_eval(True)
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            gaussians.set_eval(False)
            if (iteration in saving_iterations):
                logging.info("[ITER {}] Saving Gaussians".format(iteration))
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if classifier is not None:
                    ckpt_dir = os.path.join(dataset.model_path, "ckpts_point_cloud", "iteration_{}".format(iteration))
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(classifier.state_dict(), os.path.join(ckpt_dir, "classifier.pth"))

            
            # Densificatio

            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])#
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)       

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold,args.opacity_threshold, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if cls_optimizer is not None:
                    cls_optimizer.step()
                    cls_optimizer.zero_grad()

    #drawing training loss curve
    fig = plt.figure()
    logging.info("Drawing Training loss curve")
    print("\nDrawing Training loss curve")
    plt.plot(record_loss)
    plt.xlabel('Epochs')
    plt.ylabel('Training loss')
    plt.title('Training error curve')
    os.makedirs(os.path.join(scene.model_path,"train", "ours_{}".format(iteration)),exist_ok=True)
    fig.savefig(os.path.join(scene.model_path,"train", "ours_{}".format(iteration),"training_loss.png"))
    #render result and evaluate metrics
    with torch.no_grad():
        if args.render_after_train:
            gaussians.set_eval(True)
            if args.scene_name!="lego":
                torch.cuda.empty_cache()
                logging.info(f"Rendering testing set [{len(scene.getTestCameras())}]...")
                render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipe, \
                background,render_multi_view=True,render_s2d_inter=True)
                if classifier is not None:
                    logging.info("Rendering test segmentation...")
                    render_segmentation_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipe, background, classifier, segmentation_palette)

                torch.cuda.empty_cache()
                logging.info(f"Rendering training set [{len(scene.getTrainCameras())}]...")
                render_set(dataset.model_path, "train", iteration, scene.getTrainCameras(), gaussians, pipe, background)
                if classifier is not None:
                    logging.info("Rendering train segmentation...")
                    render_segmentation_set(dataset.model_path, "train", iteration, scene.getTrainCameras(), gaussians, pipe, background, classifier, segmentation_palette)

                if gaussians.color_net_type in ["naive"]:
                    logging.info(f"Rendering training set's intrinsic image [{len(scene.getTrainCameras())}]...")
                    
                    render_intrinsic(dataset.model_path, "train", iteration, scene.getTrainCameras(), gaussians, pipe, background)
                    
                        
                logging.info(f"Test rendering speed [{len(scene.getTrainCameras())}]...")
                avg_rendering_speed=test_rendering_speed( scene.getTrainCameras(), gaussians, pipe, background)
                logging.info(f"rendering speed:{avg_rendering_speed}s/image")
                if gaussians.color_net_type in ["naive"]:
                    logging.info(f"Test rendering speed using cache [{len(scene.getTrainCameras())}]...")
                    avg_rendering_speed=test_rendering_speed( scene.getTrainCameras(), gaussians, pipe, background,use_cache=True)
                    logging.info(f"rendering speed using cache:{avg_rendering_speed}s/image")
                
                
                if args.metrics_after_train :
                    logging.info("Evaluating metrics on testing set...")
                    evaluate_metrics([dataset.model_path],use_logs=True)
                    logging.info("Evaluating metrics half image on testing set...")
                    evaluate_metrics_half([dataset.model_path],use_logs=True)
                    
            elif args.scene_name=="lego":
                
                torch.cuda.empty_cache()
                logging.info(f"Rendering testing set [{len(scene.getTestCameras())}]...")
                render_lego(dataset.model_path, "test", iteration,  scene.getTestCameras(),scene.getTrainCameras()[0], gaussians, pipe, background)
                
                if args.metrics_after_train :
                    logging.info("Evaluating metrics on testing set...")
                    evaluate_metrics([dataset.model_path],use_logs=True)
                    logging.info("Evaluating metrics half image on testing set...")
                    evaluate_metrics_half([dataset.model_path],use_logs=True)
            gaussians.set_eval(False)
            
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations and scene.scene_name!="lego":
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 500, 25)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                logging.info("[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        
    elif iteration in testing_iterations and scene.scene_name=="lego":
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()},)
        assert scene.gaussians.color_net_type in ["naive"],"color_net_type should be naive"
        rendering = render(scene.getTrainCameras()[0], scene.gaussians, pipe=renderArgs[0], bg_color=renderArgs[1],store_cache=True)["render"]
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, pipe=renderArgs[0], bg_color=renderArgs[1],use_cache=True)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                logging.info("[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i*5000 for i in range(1,20)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--render_after_train",  action='store_true', default=True)
    parser.add_argument("--metrics_after_train",  action='store_true', default=True)
    parser.add_argument("--data_perturb", nargs="+", type=str, default=[])#for lego ["color","occ"]

    # Gaussian-grouping segmentation arguments
    parser.add_argument("--object_path", type=str, default="", help="Directory of per-image SAMD segmentation masks")
    parser.add_argument("--reg3d_interval", type=int, default=2)
    parser.add_argument("--reg3d_k", type=int, default=5)
    parser.add_argument("--reg3d_lambda_val", type=float, default=2.0)
    parser.add_argument("--reg3d_max_points", type=int, default=300000)
    parser.add_argument("--reg3d_sample_size", type=int, default=1000)

    args = parser.parse_args(sys.argv[1:])         
    args.save_iterations.append(args.iterations)     
    args=args_init.argument_init(args)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    op.position_lr_max_steps=op.iterations
            
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.debug_from,args)
    
    # All done
    print("\nTraining complete.")
