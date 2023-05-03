import argparse
import datetime
import random
import time
from pathlib import Path

import torch
import torchvision.transforms as standard_transforms
import numpy as np

from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
from glob import glob
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    
    parser.add_argument('--dataset', default='jhu', type=str,
                        help="name of dataset")
    
    # parser.add_argument('--root', default='/home/ubuntu/workspace/bekhzod/DM-Count/datasets/jhu_crowd_v2.0/test', type=str,
    #                     help="name of dataset")
    
    parser.add_argument('--root', default='drone_ims', type=str,
                        help="name of dataset")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--output_dir', default='',
                        help='path where to save')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=2, type=int, help='the gpu used for evaluation')

    return parser

@torch.no_grad()
def main(args, debug = False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model(args, device)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # im_paths = sorted(glob(f"{args.root}/images/*.jpg"))[:50]
    im_paths = sorted(glob(f"{args.root}/*.jpg"))[:50]
    # gts = [int(line.split(",")[1]) for line in open(f"{args.root}/image_labels.txt", 'r').readlines()]
    save_path = f"{args.output_dir}/{args.weight_path.split('/')[0]}/{args.dataset}_{os.path.basename(args.root)}" 
    os.makedirs(save_path, exist_ok=True)
    mses = []
    
    for idx, img_path in enumerate(im_paths):
        # if idx == 2: break
        im_name = os.path.basename(img_path)
                
        # load the images
        img_raw = Image.open(img_path).convert('RGB')
        # round the size
        width, height = img_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        # pre-proccessing
        img = transform(img_raw)

        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(device)
        # run inference
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]

        threshold = 0.5
        # filter the predictions
        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        # mse = (gts[idx] - predict_cnt) ** 2
        # mses.append(mse)
        
        # print(f"\nGT Number of People in {im_name} -> {gts[idx]}")
        # print(f"Predicted Number of People -> {predict_cnt}")
        # print(f"MSE -> {mse}\n")

        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

        outputs_points = outputs['pred_points'][0]
        # draw the predictions
        size = 2
        img_to_draw = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)
        for p in points:
            img_to_draw = cv2.circle(img_to_draw, (int(p[0]), int(p[1])), size, (0, 255, 0), -1) # 0,0,255 for red
        # save the visualized image
        cv2.imwrite(os.path.join(save_path, f'{os.path.basename(img_path)}_out_img_{os.path.basename(args.weight_path)}_{predict_cnt}.jpg'), img_to_draw)
    
    print(f"Mean MSE for the whole dataset: {np.mean(mses)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
