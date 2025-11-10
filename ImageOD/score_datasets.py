######################################################################################################################
# This file contains code adapted from https://github.com/Jingkang50/OpenOOD
#                   by "OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection" (Zhang et. al 2024)
#
# Additional implementations have been added to the file
######################################################################################################################

import numpy as np
import argparse
import os
import sys
sys.path.append('./openood')
from ImageOD.c_eval import cEvaluator
from openood.networks import ResNet18_32x32
from ImageOD.models.resnet import resnet20
from collections import OrderedDict
import torch
#from config_model import config
#model = config()
dataset2classes = {'cifar10': 10, 'cifar100': 100, 'imagenet': 1000}
net_name2object = {'resnet18_32x32': ResNet18_32x32, 'resnet20_32x32': resnet20}


"""
def get_network(net_name, net_ckpt_path, id_dataset):
    net = net_name2object[net_name](num_classes=dataset2classes[id_dataset])
    #net.load_state_dict(torch.load(net_ckpt_path))
    state_dict = torch.load(net_ckpt_path, map_location="cuda")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    net.load_state_dict(new_state_dict)
    net.cuda()
    net.eval()
    return net
"""
def get_network(net_name, net_ckpt_path, id_dataset):
    net = net_name2object[net_name](num_classes=dataset2classes[id_dataset])

    ckpt = torch.load(net_ckpt_path, map_location="cuda")
    # handle checkpoint formats
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("module.", "")  # strip prefixes
        new_state_dict[new_k] = v

    net.load_state_dict(new_state_dict, strict=True)
    net.cuda()
    net.eval()
    return net

def save_scores(args):
    # === Landseer I/O handling ===
    if os.path.exists(args.input_dir):
        print(f"[RCOD] Using preprocessed Landseer data from {args.input_dir}")
        X_train = np.load(os.path.join(args.input_dir, "data.npy"))
        Y_train = np.load(os.path.join(args.input_dir, "labels.npy"))
        X_test = np.load(os.path.join(args.input_dir, "test_data.npy"))
        Y_test = np.load(os.path.join(args.input_dir, "test_labels.npy"))
    else:
        print(f"[RCOD] No input-dir found at {args.input_dir}, using default datasets")

    args.save_path = args.output
    # load the model
    net = get_network(args.net, args.net_ckpt_path, args.id_dataset)
    evaluator = cEvaluator(
        net,
        id_name=args.id_dataset,
        data_root='./openood/data',
        config_root=None,
        preprocessor=None,
        postprocessor_name=args.postprocess,
        postprocessor=None,
        batch_size=200,
        shuffle=False,
        num_workers=2,
        n_train=args.n_train,
        p_train=args.p_train,
        curr_ood=args.ood_dataset,
        )

    if args.id_dataset == 'imagenet':
        fsood = True
    else:
        fsood = False
    evaluator.eval_ood(fsood=fsood, curr_ood=args.ood_dataset)
    # save scores to files
    save_path = os.path.join(args.save_path, args.id_dataset, args.net, args.postprocess)
    try:
        os.makedirs(save_path)
    except:
        pass
    id_scores = evaluator.scores['id']['test']
    id_file = os.path.join(save_path, args.id_dataset + '_test.npy')
    print(f'saving {id_file} ...')
    np.save(id_file, id_scores)
    for ood_type in ['near', 'far']:
        ood_scores = evaluator.scores['ood'][ood_type]
        for ood_dataset in ood_scores.keys():
            if ood_dataset == args.ood_dataset:
                ood_file = os.path.join(save_path, ood_dataset + '.npy')
                print(f'saving {ood_file} ...')
                np.save(ood_file, ood_scores[ood_dataset])


def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--save_path', type=str, default=None, help='Path to save scores files.')
    parser.add_argument("--input-dir", type=str, default="/data", help="Directory containing preprocessed .npy inputs.")
    parser.add_argument("--output", type=str, default="/output", help="Directory to save results.")
    parser.add_argument("--save_path", type=str, default=None, help="(Deprecated) Path to save scores files.")
    parser.add_argument('--net', type=str, default=None, choices=['resnet18_32x32', 'resnet20_32x32'])
    parser.add_argument('--net_ckpt_path', type=str, default=None, help='Path to checkpoint file.')
    parser.add_argument('--postprocess', type=str, default=None, choices=["react"])
    parser.add_argument('--id_dataset', type=str, default=None, choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--ood_dataset', type=str, default=None)
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--p_train', type=float, default=0.03)
    args = parser.parse_args()
    args.save_path = args.output
        #raise ValueError('Save path must be specified.')
    if not os.path.isfile(args.net_ckpt_path):
        raise ValueError('Checkpoint file path does not exsit.')
    return args


if __name__ == "__main__":
    args = get_args()
    save_scores(args)

