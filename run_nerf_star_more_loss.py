import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

# import matplotlib.pyplot as plt

from run_nerf_helpers import *

from load_blender import load_blender_data_star

import configargparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def config_parser():

    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    parser.add_argument("--ft_path_trained", type=str, default=None, 
                        help='trained model from the internet')
    parser.add_argument("--beta", type=float, default=2e-3, 
                        help='regularization coefficient')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    parser.add_argument("--render_time", type=int, default=1, 
                        help='timestep to render the scene at')
    parser.add_argument("--cam_number", type=int, default=1, 
                        help='camera to render the scene for all times')
    parser.add_argument("--render_along_time", action='store_false', 
                        help='set true for rending at fix cam position')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=4, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')

    parser.add_argument("--N_iters",   type=int, default=50000, 
                        help='total number of training iterations')

    return parser

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_nerf_star(args):
    """Instantiate STaR's 2 NeRF MLP models.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model_s = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    model_d = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model_s.parameters())
    grad_vars += list(model_d.parameters())

    model_s_fine = None
    model_s_fine = None
    if args.N_importance > 0:
        model_s_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        model_d_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_s_fine.parameters())
        grad_vars += list(model_d_fine.parameters())

    #### TODO: maybe edit this 
    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # assert args.ft_path_trained is not None, 'check usage please'
    # ckpt_trained_path = os.path.join(basedir, expname, args.ft_path_trained + '.tar')

    # if args.ft_path is not None and args.ft_path!='None':
    #     ckpts = [args.ft_path]
    # else:
    #     ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if '0.tar' in f]

    # print('Found ckpts', ckpts)
    # print('Found our super trianed model', ckpt_trained_path)

    # if len(ckpts) > 0 and not args.no_reload:
    #     ckpt_path = ckpts[-1]
    #     print('Reloading from', ckpt_path)
    #     ckpt = torch.load(ckpt_path)

    #     ckpt_trained = torch.load(ckpt_trained_path)
        
    #     start = ckpt['global_step']
    #     # optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    #     # Load model
    #     model_s.load_state_dict(ckpt_trained['network_fn_state_dict'])
    #     model_d.load_state_dict(ckpt['network_fn_state_dict_d'])
    #     if model_s_fine is not None:
    #         model_s_fine.load_state_dict(ckpt_trained['network_fine_state_dict'])
    #     if model_d_fine is not None:
    #         model_d_fine.load_state_dict(ckpt['network_fine_state_dict_d'])
    
    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if '0.tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model_s.load_state_dict(ckpt['network_fn_state_dict_s'])
        model_d.load_state_dict(ckpt['network_fn_state_dict_d'])
        if model_s_fine is not None:
            model_s_fine.load_state_dict(ckpt['network_fine_state_dict_s'])
        if model_d_fine is not None:
            model_d_fine.load_state_dict(ckpt['network_fine_state_dict_d'])

    ##########################

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine_s' : model_s_fine,
        'network_fine_d' : model_d_fine,
        'N_samples' : args.N_samples,
        'network_fn_s' : model_s,
        'network_fn_d' : model_d,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # torch.save({
    #     # 'global_step': 1000,
    #     # 'network_fn_state_dict_s': render_kwargs_train['network_fn_s'].state_dict(),
    #     # 'state_dict': render_kwargs_train['network_fine_s'].state_dict(),
    #     # 'network_fn_state_dict_d': render_kwargs_train['network_fn_d'].state_dict(),
    #     'state_dict': render_kwargs_train['network_fine_d'].state_dict(),
    #     # 'optimizer_state_dict': optimizer.state_dict(),
    # }, './old_format_ball_bad.tar', _use_new_zipfile_serialization=False)
    # print('Saved checkpoints at ./old_format_ball_bad.tar')
    # sys.exit()

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def raw2outputs(raw_stat, raw_dyna, z_vals, rays_d_s, rays_d_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    # print(z_vals)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    # print(torch.norm(rays_d_s[...,None,:], dim=-1))
    # print(torch.norm(rays_d_d[...,None,:], dim=-1))
    # print(dists)

    # print(rays_d_s.shape)
    # print((torch.norm(rays_d_s[...,None,:], dim=-1).shape))
    

    assert torch.allclose(torch.norm(rays_d_s[...,None,:], dim=-1), torch.norm(rays_d_d[...,None,:], dim=-1)), 'wrong understanding of rotation matrix or torch!!!!! In raw2outputs'
    
    dists = dists * torch.norm(rays_d_s[...,None,:], dim=-1)

    rgb_stat = torch.sigmoid(raw_stat[...,:3])  # [N_rays, N_samples, 3]
    rgb_dyna = torch.sigmoid(raw_dyna[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_stat[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw_stat[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)
    
    alpha_s = raw2alpha(raw_stat[...,3] + noise, dists)  # [N_rays, N_samples]
    alpha_d = raw2alpha(raw_dyna[...,3] + noise, dists)
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    T_s = torch.cumprod(torch.cat([torch.ones((alpha_s.shape[0], 1)), 1.-alpha_s + 1e-10], -1), -1)[:, :-1]
    T_d = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), 1.-alpha_d + 1e-10], -1), -1)[:, :-1]
    # weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    T = T_s * T_d
    weights_stat = T * alpha_s
    weights_dyna = T * alpha_d
    rgb_map = torch.sum(weights_stat[...,None] * rgb_stat, -2)  # [N_rays, 3]
    rgb_map += torch.sum(weights_dyna[...,None] * rgb_dyna, -2)

    weights = weights_stat + weights_dyna
    depth_map = torch.sum(weights * z_vals, -1)

    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    ##### static model only
    weights_stat_only = T_s * alpha_s
    rgb_map_stat = torch.sum(weights_stat_only[...,None] * rgb_stat, -2)  # [N_rays, 3]
    depth_map_stat = torch.sum(weights_stat_only * z_vals, -1)
    disp_map_stat = 1./torch.max(1e-10 * torch.ones_like(depth_map_stat), depth_map_stat / torch.sum(weights_stat_only, -1))
    acc_map_stat = torch.sum(weights_stat_only, -1)
    if white_bkgd:
        rgb_map_stat = rgb_map_stat + (1.-acc_map_stat[...,None])
    model_outputs_stat = {
        'weights_stat': weights_stat_only,
        'rgb_stat': rgb_map_stat,
        'depth_stat': depth_map_stat,
        'disp_stat': disp_map_stat,
        'acc_stat': acc_map_stat,
    }

    ##### dynamic model only
    weights_dyna_only = T_d * alpha_d
    rgb_map_dyna = torch.sum(weights_dyna_only[...,None] * rgb_dyna, -2)  # [N_rays, 3]
    depth_map_dyna = torch.sum(weights_dyna_only * z_vals, -1)
    disp_map_dyna = 1./torch.max(1e-10 * torch.ones_like(depth_map_dyna), depth_map_dyna / torch.sum(weights_dyna_only, -1))
    acc_map_dyna = torch.sum(weights_dyna_only, -1)
    if white_bkgd:
        rgb_map_dyna = rgb_map_dyna + (1.-acc_map_dyna[...,None])
    model_outputs_dyna = {
        'weights_dyna': weights_dyna_only,
        'rgb_dyna': rgb_map_dyna,
        'depth_dyna': depth_map_dyna,
        'disp_dyna': disp_map_dyna,
        'acc_dyna': acc_map_dyna,
    }

    #### losses terms
    
    # print(alpha_s.shape)
    # print(alpha_s)
    # # assert torch.all(alpha_s<=1.0), 'alpha s is problem 1'
    # assert torch.all(alpha_s>=0.0), 'alpha s is problem 0 '
    
    # assert torch.all(alpha_d>=0.0) and torch.all(alpha_d<=1.0), 'alpha d is problem'

    H_alpha_s = alpha_s * torch.log(alpha_s + 1e-10) + (1 - alpha_s) * torch.log(1 - alpha_s + 1e-10)
    H_alpha_d = alpha_d * torch.log(alpha_d + 1e-10) + (1 - alpha_d) * torch.log(1 - alpha_d + 1e-10)
    alpha_s_d_sum = alpha_s + alpha_d
    
    H_s_d = alpha_s * torch.log(alpha_s + 1e-10) + alpha_d * torch.log(alpha_d + 1e-10) - alpha_s_d_sum * torch.log(alpha_s_d_sum + 1e-10)
    
    reg_loss = (H_alpha_s + H_alpha_d + H_s_d)
    
    return rgb_map, disp_map, acc_map, weights, depth_map, model_outputs_stat, model_outputs_dyna, reg_loss


def render_rays(ray_batch_stat, ray_batch_dyna,
                network_fn_s, network_fn_d,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine_s=None, network_fine_d=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch_stat.shape[0]
    rays_o_s, rays_d_s = ray_batch_stat[:,0:3], ray_batch_stat[:,3:6] # [N_rays, 3] each
    rays_o_d, rays_d_d = ray_batch_dyna[:,0:3], ray_batch_dyna[:,3:6] # [N_rays, 3] each
    viewdirs_s = ray_batch_stat[:,-3:] if ray_batch_stat.shape[-1] > 8 else None
    viewdirs_d = ray_batch_dyna[:,-3:] if ray_batch_dyna.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch_stat[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    # print(near)
    # print(far)

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts_stat = rays_o_s[...,None,:] + rays_d_s[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    pts_dyna = rays_o_d[...,None,:] + rays_d_d[...,None,:] * z_vals[...,:,None]

#     raw = run_network(pts)
    raw_stat = network_query_fn(pts_stat, viewdirs_s, network_fn_s)
    raw_dyna = network_query_fn(pts_dyna, viewdirs_d, network_fn_d)

    mask = torch.any((pts_dyna <= -0.15) | (pts_dyna >= 0.15) , 2)
    # print(mask.shape)
    raw_dyna[mask] = torch.Tensor([0.0,0.0,0.0,0.0])
    
    rgb_map, disp_map, acc_map, weights, depth_map, model_outputs_stat, model_outputs_dyna, reg_loss = raw2outputs(raw_stat, raw_dyna, z_vals, rays_d_s, rays_d_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts_stat = rays_o_s[...,None,:] + rays_d_s[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]
        pts_dyna = rays_o_d[...,None,:] + rays_d_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn_s = network_fn_s if network_fine_s is None else network_fine_s
        run_fn_d = network_fn_d if network_fine_d is None else network_fine_d
#         raw = run_network(pts, fn=run_fn)

        raw_stat = network_query_fn(pts_stat, viewdirs_s, run_fn_s)
        raw_dyna = network_query_fn(pts_dyna, viewdirs_d, run_fn_d)

        mask = torch.any((pts_dyna <= -0.15) | (pts_dyna >= 0.15) , 2)
        # print(mask.shape)
        raw_dyna[mask] = torch.Tensor([0.0,0.0,0.0,0.0])

        rgb_map, disp_map, acc_map, weights, depth_map, model_outputs_stat, model_outputs_dyna, reg_loss = raw2outputs(raw_stat, raw_dyna, z_vals, rays_d_s, rays_d_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}

    ret.update(model_outputs_stat)
    ret.update(model_outputs_dyna)
    ret['reg_loss'] = reg_loss

    if retraw:
        ret['raw_stat'] = raw_stat
        ret['raw_dyna'] = raw_dyna
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def batchify_rays(rays_flat_s, rays_flat_d, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat_s.shape[0], chunk):
        ret = render_rays(rays_flat_s[i:i+chunk], rays_flat_d[i:i+chunk],**kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, K, chunk=1024*32, rays_stat=None, rays_dyna=None, c2w=None, w2o = None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      o2w: array of shape [3, 4]. Object-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        rays_o_s, rays_d_s = get_rays(H, W, K, c2w)
        rays_o_d = transform_rays_to_object_frame(rays_o_s, w2o)
        rays_d_d = transform_rays_to_object_frame(rays_d_s, w2o, direction=True)
    else:
        # use provided ray batch
        # print('HHHEEEEEREEE')
        rays_o_s, rays_d_s = rays_stat
        rays_o_d, rays_d_d = rays_dyna

    if use_viewdirs:
        # provide ray directions as input
        viewdirs_s = rays_d_s
        viewdirs_d = rays_d_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o_s, rays_d_s = get_rays(H, W, K, c2w_staticcam)
            rays_o_d = transform_rays_to_object_frame(rays_o_s, w2o)
            rays_d_d = transform_rays_to_object_frame(rays_d_s, w2o, direction=True)
        viewdirs_s = viewdirs_s / torch.norm(viewdirs_s, dim=-1, keepdim=True)
        viewdirs_s = torch.reshape(viewdirs_s, [-1,3]).float()
        viewdirs_d = viewdirs_d / torch.norm(viewdirs_d, dim=-1, keepdim=True)
        viewdirs_d = torch.reshape(viewdirs_d, [-1,3]).float()

    sh = rays_d_s.shape # [..., 3]
    
    if ndc:
        # for forward facing scenes
        print('NDC What the Hell!!!!!!!!')
        return 
        # rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o_s = torch.reshape(rays_o_s, [-1,3]).float()
    rays_d_s = torch.reshape(rays_d_s, [-1,3]).float()
    rays_o_d = torch.reshape(rays_o_d, [-1,3]).float()
    rays_d_d = torch.reshape(rays_d_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d_s[...,:1]), far * torch.ones_like(rays_d_s[...,:1])
    rays_stat = torch.cat([rays_o_s, rays_d_s, near, far], -1)
    rays_dyna = torch.cat([rays_o_d, rays_d_d, near, far], -1)
    # print(rays_stat.shape, 'before batchify')
    if use_viewdirs:
        rays_stat = torch.cat([rays_stat, viewdirs_s], -1)
        rays_dyna = torch.cat([rays_dyna, viewdirs_d], -1)

    # Render and reshape
    # print(rays_stat.shape, 'in batchify')
    all_ret = batchify_rays(rays_stat, rays_dyna, chunk, **kwargs)
    # all_ret_d = batchify_rays(rays_d, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(render_poses, render_object_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    rgbs_stat = []
    disps_stat = []
    rgbs_dyna = []
    disps_dyna = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        w2o = render_object_poses[i]
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, ret_dict = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], w2o = w2o[:3,:4], **render_kwargs)
        
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        
        rgbs_stat.append(ret_dict['rgb_stat'].cpu().numpy())
        disps_stat.append(ret_dict['disp_stat'].cpu().numpy())
        
        rgbs_dyna.append(ret_dict['rgb_dyna'].cpu().numpy())
        disps_dyna.append(ret_dict['disp_dyna'].cpu().numpy())

        if i==0:
            print(rgb.shape, disp.shape)

        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

            rgb8_stat = to8b(rgbs_stat[-1])
            filename_stat = os.path.join(savedir, 'stat_{:03d}.png'.format(i))
            imageio.imwrite(filename_stat, rgb8_stat)

            rgb8_dyna = to8b(rgbs_dyna[-1])
            filename_dyna = os.path.join(savedir, 'dyna_{:03d}.png'.format(i))
            imageio.imwrite(filename_dyna, rgb8_dyna)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    rgbs_stat = np.stack(rgbs_stat, 0)
    disps_stat = np.stack(disps_stat, 0)

    rgbs_dyna = np.stack(rgbs_dyna, 0)
    disps_dyna = np.stack(disps_dyna, 0)

    return rgbs, rgbs_stat, rgbs_dyna, disps, disps_stat, disps_dyna

# def render_path(render_poses, render_object_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

#     H, W, focal = hwf

#     if render_factor!=0:
#         # Render downsampled for speed
#         H = H//render_factor
#         W = W//render_factor
#         focal = focal/render_factor

#     rgbs = []
#     disps = []

#     t = time.time()
#     for i, c2w in enumerate(tqdm(render_poses)):
#         w2o = render_object_poses[i]
#         print(i, time.time() - t)
#         t = time.time()
#         rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], w2o = w2o[:3,:4], **render_kwargs)
#         rgbs.append(rgb.cpu().numpy())
#         disps.append(disp.cpu().numpy())
#         if i==0:
#             print(rgb.shape, disp.shape)

#         """
#         if gt_imgs is not None and render_factor==0:
#             p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
#             print(p)
#         """

#         if savedir is not None:
#             rgb8 = to8b(rgbs[-1])
#             filename = os.path.join(savedir, '{:03d}.png'.format(i))
#             imageio.imwrite(filename, rgb8)


#     rgbs = np.stack(rgbs, 0)
#     disps = np.stack(disps, 0)

#     return rgbs, disps

def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    # time_sequence = np.arange(15)
    # cam_sequence = np.ones(15) * 3
    # r_a_t = False
    
    time_sequence = None
    cam_sequence = None
    r_a_t = True

    images, poses, obj_poses, render_poses, render_obj_pose, hwf, i_split = load_blender_data_star(args.datadir, args.half_res, args.testskip, 
                                                    time = time_sequence, cam=cam_sequence, r_a_t = r_a_t)
    
    # images, poses, obj_poses, render_poses, render_obj_pose, hwf, i_split = load_blender_data_star(args.datadir, args.half_res, args.testskip, 
    #                                                 time = args.render_time, cam_number = args.cam_number, r_a_t = args.render_along_time)
    
    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split

    near = 2.
    far = 6.

    if args.white_bkgd:
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    else:
        images = images[...,:3]

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])
        render_obj_pose = np.array(obj_poses[i_test]) 

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    
    # # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf_star(args)
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move testing data to GPU
    render_poses = torch.Tensor(render_poses).to(device)
    render_obj_pose = torch.Tensor(render_obj_pose).to(device)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[i_test]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format('test_w_bound' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            #### TODO: need variations along time or space or 
            rgbs, rgbs_stat, rgbs_dyna, disps, disps_stat, disps_dyna = render_path(render_poses, render_obj_pose, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'stat_video.mp4'), to8b(rgbs_stat), fps=30, quality=8)
            imageio.mimwrite(os.path.join(testsavedir, 'dyna_video.mp4'), to8b(rgbs_dyna), fps=30, quality=8)

            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    # use_batching = not args.no_batching
    use_batching = True
    if use_batching:
        # For random ray batching
        print('get rays')
        rays_stat = []
        rays_dyna = []
        for i,p in enumerate(poses[:,:3,:4]):
            w2o = (obj_poses[i])[:3,:4]
            rays_o_s, rays_d_s = get_rays_np(H, W, K, p)
            # print(rays_o_s.shape)
            rays_o_d = transform_rays_to_object_frame_np(rays_o_s, w2o)
            rays_d_d = transform_rays_to_object_frame_np(rays_d_s, w2o, direction=True)
            # print(w2o)
            # print(np.isclose(np.linalg.norm(rays_d_d,axis =2), np.linalg.norm(rays_d_s,axis =2)))
            rays_stat.append((rays_o_s, rays_d_s))
            rays_dyna.append((rays_o_d, rays_d_d))
        rays_stat = np.stack(rays_stat, 0)
        rays_dyna = np.stack(rays_dyna, 0)
        # rays_stat = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        
        print('done, concats')
        rays_rgb_stat = np.concatenate([rays_stat, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb_stat = np.transpose(rays_rgb_stat, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb_stat = np.stack([rays_rgb_stat[i] for i in i_train], 0) # train images only
        rays_rgb_stat = np.reshape(rays_rgb_stat, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb_stat = rays_rgb_stat.astype(np.float32)
        
        rays_rgb_dyna = np.concatenate([rays_dyna, images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        rays_rgb_dyna = np.transpose(rays_rgb_dyna, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        rays_rgb_dyna = np.stack([rays_rgb_dyna[i] for i in i_train], 0) # train images only
        rays_rgb_dyna = np.reshape(rays_rgb_dyna, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb_dyna = rays_rgb_dyna.astype(np.float32)
        
        print('shuffle rays')
        idx = np.arange(rays_rgb_stat.shape[0])
        np.random.shuffle(idx)
        rays_rgb_stat = rays_rgb_stat[idx]
        rays_rgb_dyna = rays_rgb_dyna[idx]
        # np.random.shuffle(rays_rgb_stat)

        print('done')
        i_batch = 0

    # Move training data to GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    obj_poses = torch.Tensor(obj_poses).to(device)
    if use_batching:
        rays_rgb_stat = torch.Tensor(rays_rgb_stat).to(device)
        rays_rgb_dyna = torch.Tensor(rays_rgb_dyna).to(device)
    
    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    start = start + 2
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            batch_s = rays_rgb_stat[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch_s = torch.transpose(batch_s, 0, 1)
            batch_rays_stat, target_s = batch_s[:2], batch_s[2]
            
            batch_d = rays_rgb_dyna[i_batch:i_batch+N_rand] # [B, 2+1, 3*?]
            batch_d = torch.transpose(batch_d, 0, 1)
            batch_rays_dyna, target_d = batch_d[:2], batch_d[2]

            i_batch += N_rand
            if i_batch >= rays_rgb_stat.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb_stat.shape[0])
                rays_rgb_stat = rays_rgb_stat[rand_idx]
                rays_rgb_dyna = rays_rgb_dyna[rand_idx]
                i_batch = 0
        else:
            print('What ar you even doing here!!!! Use batching please')
        
        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays_stat=batch_rays_stat, 
                                                rays_dyna=batch_rays_dyna,
                                                verbose=i < 10, retraw=True,
                                                **render_kwargs_train)

        ### TODO: Add entropy loss

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        # trans = extras['raw'][...,-1]
        reg_loss = torch.mean(extras['reg_loss'])
        loss = img_loss + args.beta * reg_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict_s': render_kwargs_train['network_fn_s'].state_dict(),
                'network_fine_state_dict_s': render_kwargs_train['network_fine_s'].state_dict(),
                'network_fn_state_dict_d': render_kwargs_train['network_fn_d'].state_dict(),
                'network_fine_state_dict_d': render_kwargs_train['network_fine_d'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if i%args.i_video==0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, rgbs_stat, rgbs_dyna, disps, disps_stat, disps_dyna = render_path(render_poses, render_obj_pose, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)

            moviebase_stat = os.path.join(basedir, expname, '{}_stat_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase_stat + 'rgb.mp4', to8b(rgbs_stat), fps=30, quality=8)

            moviebase_dyna = os.path.join(basedir, expname, '{}_dyna_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase_dyna + 'rgb.mp4', to8b(rgbs_dyna), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i%args.i_testset==0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device),torch.Tensor(obj_poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item(), reg_loss.item()}  PSNR: {psnr.item()}")

        global_step += 1

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
