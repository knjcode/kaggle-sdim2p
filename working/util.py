import math
import time
import torch
import numpy as np
import pandas as pd

from argparse import ArgumentParser


class DummySummaryWriter:
    def __init__(*args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, *args, **kwargs):
        return self


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def save_checkpoint(conf, model, save_filename):
    if conf.torch_compile:
        model = model._orig_mod
    ckpt = {
        'model': model.state_dict()
    }
    torch.save(ckpt, save_filename)


def get_dataset_and_feats(conf):
    base_dir = conf.base_dir
    dataset_name = conf.dataset_name

    datasets = dataset_name.split(',')

    if len(datasets) > 1:
        df_list = []
        feat_list = []
        for dataset_name in datasets:
            dataset_file = f"{base_dir}{dataset_name}.tsv"
            dataset = pd.read_table(dataset_file, low_memory=False)
            embedding_file = f"{base_dir}{dataset_name}.npy"
            feats = np.load(embedding_file)

            df_list.append(dataset)
            feat_list.append(feats)

        dataset = pd.concat(df_list, axis=0).reset_index(drop=True)
        feats = np.concatenate(feat_list, axis=0)
    else:
        dataset_file = f"{base_dir}{dataset_name}.tsv"
        dataset = pd.read_table(dataset_file, low_memory=False)
        embedding_file = f"{base_dir}{dataset_name}.npy"
        feats = np.load(embedding_file)

    return dataset, feats


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--conf', required=True,
                        help='Path to config file')
    parser.add_argument('-m', '--ckpt_path', type=str, default=None)
    parser.add_argument('--amp_dtype', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--valid_bs', type=int, default=None)
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--save_feats', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--val_fold', type=int, default=None)
    parser.add_argument('--skip_val', action="store_true")
    parser.add_argument('--extract_feat', action="store_true")
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--backend', type=str, default='ddp')

    return parser.parse_args()

