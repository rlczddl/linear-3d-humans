import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import trimesh
from PIL import Image
from pathlib import Path
import argparse
import json
import pyrender
import random

import smplx

from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import colors, makepath
#from human_body_prior.mesh.mesh_viewer import MeshViewer
from mesh_viewer import MeshViewer
from human_body_prior.tools.visualization_tools import imagearray2file, smpl_params2ply


MODELS_DIR = 'models/'
VPOSER_DIR = 'vposer_v1_0/'
DATA_DIR = 'data/'
IMG_DIR_TEMPLATE = os.path.join(DATA_DIR, '{}/imgs/')
GT_DIR_TEMPLATE = os.path.join(DATA_DIR, '{}/gt/')

GENDER_DICT = { 
        'male': 0,
        'female': 1,
        'neutral': 2
        }

#os.environ['PYOPENGL_PLATFORM'] = 'egl' # Uncommnet this line while running remotely


def render_sample(vertices, faces, dataset_name, gender, subject_idx, pose_idx):
#    img_dir = os.path.join(
#            IMG_DIR_TEMPLATE.format(dataset_name), f'{gender}{subject_idx:04d}')
    img_dir = IMG_DIR_TEMPLATE.format(dataset_name)
#    os.makedirs(os.path.join(img_dir, '0'), exist_ok=True)
    os.makedirs(os.path.join(img_dir), exist_ok=True)

    rot_step = random.randint(0, 360)
    imw, imh = 600, 600
    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    body_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, 
            vertex_colors=np.tile(colors['grey'], (6890, 1)))
#    apply_mesh_tranfsormations_([body_mesh],
#                                trimesh.transformations.rotation_matrix(
#                                    np.radians(rot_step), (0, 1, 0)))
    mv.set_meshes([body_mesh], group_name='static')

    image = mv.render()
    image = Image.fromarray(image, 'RGB')
#    img_path = os.path.join(img_dir, '0', f'{pose_idx:08d}.png')
    img_path = os.path.join(img_dir, f'{subject_idx:08d}.png')
    image.save(img_path)

# TODO: START IDX
def save_points(points, dataset_name, points_type, subject_idx, start_idx):
#    joint_dir = os.path.join(
#            GT_DIR_TEMPLATE.format(dataset_name), f'{gender}{subject_idx:04d}')
    joint_dir = GT_DIR_TEMPLATE.format(dataset_name)
    Path(joint_dir).mkdir(parents=True, exist_ok=True)
#    np.save(os.path.join(joint_dir, f'{pose_idx:08d}.npy'), joints)
    np.save(os.path.join(joint_dir, 
        f'{points_type}{subject_idx+start_idx:08d}.npy'), points)


def generate_sample(dataset_name, gender, model, shape_coefs, body_pose, 
        sid, pid, start_idx, render=False):
    model.register_parameter('body_pose', 
            nn.Parameter(body_pose, requires_grad=False))

    output = model(betas=shape_coefs, return_verts=True)

    joints = output.joints.detach().cpu().numpy().squeeze()
    vertices = output.vertices.detach().cpu().numpy().squeeze()

    if render:
#        vertices = output.vertices.detach().cpu().numpy().squeeze()
        faces = model.faces.squeeze()
        render_sample(vertices, faces, dataset_name, gender, 
                sid, pid)
    save_points(joints, dataset_name, 'jnt', sid, start_idx)
    save_points(vertices, dataset_name, 'vrt', sid, start_idx)


def create_model(gender, init_body_pose, num_coefs=10):
    return smplx.create(MODELS_DIR, model_type='smplx',
                        gender=gender, use_face_contour=False,
                        num_betas=num_coefs,
                        body_pose=init_body_pose,
                        ext='npz')


def save_params(dataset_name, gender, sid, shape_coefs):
    save_dir = os.path.join(GT_DIR_TEMPLATE.format(dataset_name), f'{gender}{sid:04d}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'params.json')
    params = {'betas': shape_coefs[sid].tolist(), 'gender': GENDER_DICT[gender]}
    with open(save_path, 'w') as fjson:
        json.dump(params, fjson)


def generate_subjects(dataset_name, gender, num_subjects, 
        body_poses, vposer, regenerate=False, render=False, num_coefs=10):

    def get_last_idx(dataset_name, gender):
#        subject_dirnames = [x for x in os.listdir(
#            GT_DIR_TEMPLATE.format(dataset_name)) if 'npy' not in x]
        gt_dir = GT_DIR_TEMPLATE.format(dataset_name)
        fnames = os.listdir(gt_dir)
        if len(fnames) > 0:
            last_idx = int(sorted(fnames)[-1][-12:-4])
        else:
            last_idx = 0
        return last_idx

    if regenerate:
        start_idx = 0
    else:
        start_idx = get_last_idx(dataset_name, gender)

    if num_subjects <= 0:
        return
#    np_shape_coefs = np.random.normal(0., 1., 
#            size=(num_subjects + start_idx, 1, num_coefs)).astype(np.float32)
    np_shape_coefs = []
#    for idx in range(num_subjects + start_idx):
    range_ = range(-2, 3, 1)
    ranges = np.stack(np.meshgrid(*tuple([range_] * 6)), -1).reshape(-1, 6)
#    for component_idx in [1, 2, 3, 4, 5, 7]:
#        range_ = range(-4, 4, 2) if component_idx != 1 else range(-3, 3, 2)
#        for coef_idx in range(7):
#            coefs = np.zeros(num_coefs)
#            coefs[component_idx] = coef
#            np_shape_coefs.append(coefs)
    np_shape_coefs = []
    for array_idx in range(start_idx, ranges.shape[0]):
        for actual_idx, component_idx in enumerate([1, 2, 3, 4, 5, 7]):
            coefs = np.zeros(num_coefs)
            coefs[component_idx] = ranges[array_idx][actual_idx]
            np_shape_coefs.append(coefs)

    np_shape_coefs = np.expand_dims(np.array(np_shape_coefs, dtype=np.float32), axis=1)
    shape_coefs = torch.from_numpy(np_shape_coefs)
    model = create_model(gender, body_poses[0])

    # NOTE: If not regenerate, last shape will still be regenerated, which is OK.
#    for subject_idx in range(start_idx, start_idx + num_subjects):
    for subject_idx in range(np_shape_coefs.shape[0]):
#        save_params(dataset_name, gender, subject_idx, np_shape_coefs)
        for pose_idx in range(body_poses.shape[0]):
            print('{} {} pose {}'.format(gender, subject_idx + start_idx, pose_idx))
            generate_sample(dataset_name,
                    gender,
                    model, 
                    shape_coefs[subject_idx], 
                    body_poses[pose_idx],
                    subject_idx, 
                    pose_idx,
                    start_idx,
                    render
            )


def sample_poses(vposer_model, num_poses, loaded_np=None, output_type='aa'):
    dtype = vposer_model.bodyprior_dec_fc1.weight.dtype
    device = vposer_model.bodyprior_dec_fc1.weight.device
    vposer_model.eval()
    if loaded_np is not None:
        return torch.tensor(loaded_np, dtype=dtype).to(device)
    with torch.no_grad():
        Zgen = torch.tensor(
                np.zeros((num_poses, vposer_model.latentD)), dtype=dtype).to(device)
    body_poses = vposer_model.decode(Zgen, output_type=output_type)
    body_poses = body_poses.reshape(num_poses, 1, -1)
    return body_poses


def main(dataset_name, num_poses, num_neutral=0, num_male=0, 
        num_female=0, regenerate=False, 
        render=False, num_coefs=10):
    vposer_model, _ = load_vposer(VPOSER_DIR, vp_model='snapshot')
    
#    poses_path = os.path.join(GT_DIR_TEMPLATE.format(dataset_name), 'poses.npy')
#    create_poses_flag = regenerate or (not os.path.exists(poses_path))
    # If poses are already generated, to not override them.
#    if create_poses_flag:
    body_poses  = sample_poses(vposer_model, num_poses)
#    gt_dir = GT_DIR_TEMPLATE.format(dataset_name)
#    Path(gt_dir).mkdir(parents=True, exist_ok=True)
#    np.save(os.path.join(gt_dir, 'poses.npy'), body_poses.cpu().detach().numpy())
#    else:
#        body_poses = torch.tensor(np.load(poses_path))

    # Create dataset dir and img/ and gt/ subdirs.
    Path(IMG_DIR_TEMPLATE.format(dataset_name)).mkdir(
            parents=True, exist_ok=True)
    os.makedirs(GT_DIR_TEMPLATE.format(dataset_name), exist_ok=True)
    
    generate_subjects(dataset_name,
            'neutral', 
            num_neutral, 
            body_poses, 
            vposer_model,
            regenerate,
            render
    )
    generate_subjects(dataset_name,
            'male', 
            num_male, 
            body_poses, 
            vposer_model,
            regenerate,
            render
    )
    generate_subjects(dataset_name,
            'female', 
            num_female, 
            body_poses, 
            vposer_model,
            regenerate,
            render
    )


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
            usage='%(prog)s [OPTIONS]...',
            description='Generates specified number of meshes \
            (subject x pose). If --regenerate, then num_poses \
            is redundant.'
            )
    parser.add_argument(
            '--name', type=str, help='dataset name')
    parser.add_argument(
            '--num_poses', type=int, default=0,
            help='# of poses per subject')
    parser.add_argument(
            '--neutral', type=int, default=0,
            help='# of neutral gender subjects')
    parser.add_argument(
            '--male', type=int, default=0,
            help='# of male subjects')
    parser.add_argument(
            '--female', type=int, default=0,
            help='# of female subjects')
    parser.add_argument(
            '--regenerate', dest='regenerate',
            action='store_true', help='regenerate pose params')
    parser.add_argument(
            '--render', dest='render',
            action='store_true', help='render poses')

    return parser


def check_args(args):
    if args.neutral <= 0 and args.male <= 0 and \
            args.female <= 0:
        raise Exception('Should set at least one gender!')
    if args.num_poses <= 0:
        raise Exception('Should generate more than 0 pose!')
    return args


if __name__ == '__main__':
    parser = init_argparse()
    args = check_args(parser.parse_args())
    print(args.render)
    main(dataset_name=args.name,
         num_poses=args.num_poses, 
         num_neutral=args.neutral,
         num_male=args.male,
         num_female=args.female,
         regenerate=args.regenerate,
         render=args.render)

