import argparse
from argparse import Namespace
import os
import numpy as np
import re

import eval_tool.immatch as immatch
from eval_tool.immatch.utils.data_io import lprint
import eval_tool.immatch.utils.fire_helper as helper
from eval_tool.immatch.utils.model_helper import parse_model_config
from model.full_model import GeoFormer as GeoFormer_

def save_keypoints(kpts1, kpts2, save_dir, image_name):
    """
    Save keypoints in .npy and .txt formats.

    :param kpts1: Keypoints from Image 1
    :param kpts2: Keypoints from Image 2
    :param save_dir: Directory where files should be saved
    :param image_name: Base name for the output file (derived from image filename)
    """
    save_path_npy = os.path.join(save_dir, f"{image_name}_keypoints.npy")
    save_path_txt = os.path.join(save_dir, f"{image_name}_keypoints.txt")

    # Concatenate keypoints for saving (each row: x1, y1, x2, y2)
    keypoints = np.hstack((kpts1, kpts2))  # Shape: (N, 4)

    # Save as .npy
    np.save(save_path_npy, keypoints)

    # Save as .txt (formatted as x1, y1, x2, y2)
    np.savetxt(save_path_txt, keypoints, fmt="%.6f", delimiter=" ")

    print(f"✅ Keypoints saved: \n  - {save_path_npy}\n  - {save_path_txt}")

def eval_fire(
        root_dir,
        config_list,
        task='homography',
        h_solver='degensac',
        ransac_thres=2,
        match_thres=None,
        odir='outputs/fire',
        save_npy=True,
        print_out=False,
        debug=False,
):
    # Init paths
    data_root = os.path.join(root_dir, 'data/datasets/FIRE')
    cache_dir = os.path.join(root_dir, odir, 'cache')
    result_dir = os.path.join(root_dir, odir, 'results', task)
    mauc = 0
    gt_dir = os.path.join(data_root, 'Ground Truth')
    im_dir = os.path.join(data_root, 'Images')
    match_pairs = [x for x in os.listdir(gt_dir) if x.endswith('.txt')
                   and not x.endswith('P37_1_2.txt')]
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

        # Iterate over methods
    for config_name in config_list:
        # Load model
        args = parse_model_config(config_name, 'fire', root_dir)
        class_name = args['class']

        # One log file per method
        log_file = os.path.join(result_dir, f'{class_name}.txt')
        log = open(log_file, 'a')
        lprint_ = lambda ms: lprint(ms, log)
        ransac_thres = args['ransac_thres']
        # Iterate over matching thresholds
        thresholds = match_thres if match_thres else [args['match_threshold']]
        lprint_(f'\n>>>> Method={class_name} Default config: {args} '
                f'Thres: {thresholds}')

        for thres in thresholds:
            args['match_threshold'] = thres  # Set to target thresholds

            # Init model
            model = immatch.__dict__[class_name](args)

            matcher = lambda im1, im2: model.match_pairs(im1, im2)

            # Init result save path (for matching results)
            result_npy = None
            if save_npy:
                result_tag = model.name
                if args['imsize'] > 0:
                    result_tag += f".im{args['imsize']}"
                if thres > 0:
                    result_tag += f'.m{thres}'
                result_npy = os.path.join(cache_dir, f'{result_tag}.npy')

            lprint_(f'Matching thres: {thres}  Save to: {result_npy}')

            # Eval on the specified task(s)
            mauc = helper.eval_fire(
                matcher,
                match_pairs,
                im_dir,
                gt_dir,
                model.name,
                task=task,
                scale_H=getattr(model, 'no_match_upscale', False),
                h_solver=h_solver,
                ransac_thres=ransac_thres,
                lprint_=lprint_,
                debug=debug,
            )
            
                # Save keypoints after matching
            for pair in match_pairs:
                # Use regex to extract "P01_1", "A03_2", "S12_5", etc.
                match = re.search(r'([PAS]\d+_\d+)_\d+', pair)  
                if match:
                    im1_name = match.group(1)  # Extracts "P01_1", "A03_2", "S12_5"
                else:
                    print(f"⚠️ Warning: Could not extract image name from {pair}")
                    continue  # Skip if it fails
                im1_path = os.path.join(im_dir, f"{im1_name}.jpg")
                im2_path = os.path.join(im_dir, f"{im1_name.replace('_1', '_2')}.jpg")  # Replace _1 with _2

                # Perform matching
                matches, kpts1, kpts2, scores = model.match_pairs(im1_path, im2_path)
                
                # Save keypoints
                save_keypoints(kpts1, kpts2, result_dir, im1_name)
            
        log.close()
        return mauc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark HPatches')
    parser.add_argument('--gpu', '-gpu', type=str, default='0')
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--odir', type=str, default='outputs/fire')
    parser.add_argument('--config', type=str, nargs='*', default=['geoformer'])
    parser.add_argument('--match_thres', type=float, nargs='*', default=None)
    parser.add_argument(
        '--task', type=str, default='homography',
        choices=['matching', 'homography', 'both']
    )
    parser.add_argument(
        '--h_solver', type=str, default='cv',
        choices=['degensac', 'cv']
    )
    parser.add_argument('--ransac_thres', type=float, default=15)
    parser.add_argument('--save_npy', action='store_true', default=True)
    parser.add_argument('--print_out', action='store_true', default=True)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    eval_fire(
        args.root_dir, args.config, args.task,
        h_solver=args.h_solver,
        ransac_thres=args.ransac_thres,
        match_thres=args.match_thres,
        odir=args.odir,
        save_npy=args.save_npy,
        print_out=args.print_out
    )


