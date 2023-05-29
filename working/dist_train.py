import datetime
import logging
import pytz
import os
import random
import time
import re

import logzero
import torch
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import warnings

from torch.cuda.amp import GradScaler
from logzero import logger
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from timm.scheduler.cosine_lr import CosineLRScheduler

from config import load_config
from util import parse_args, save_checkpoint
from optimizer import get_optimizer
from scheduler import get_scheduler
from module import (
    train_fn_emb_ddp,
    valid_fn_emb_ddp,
)
from dataset import get_dataset
from model import (
    get_model,
)
from util import DummySummaryWriter, get_dataset_and_feats

import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from torch.nn.parallel import DistributedDataParallel as DDP

warnings.simplefilter('ignore', UserWarning)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class ReduceLROnPlateauPatch(ReduceLROnPlateau):
    def get_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]


def train_loop(conf, hvd, rank, local_rank, world_size, dataset, feats, fold, logger, save_dir, log_basename, log_path, total_epochs):

    seed_everything(conf.seed)
    torch.manual_seed(conf.seed)

    if conf.train:
        logger.info(f"=============== fold{fold} training ===============")
    else:
        logger.info(f"=============== fold{fold} testing ===============")

    # ckptのファイル名のfoldを置換
    if re.search('fold._best', conf.ckpt_path):
        conf.ckpt_path = re.sub('fold._best', f"fold{fold}_best", conf.ckpt_path)
        logger.info(f"replace ckpt_path: {conf.ckpt_path}")

    tb_logname = os.path.join(save_dir, f"{log_basename}")
    if rank == 0:
        tb_writer = SummaryWriter(log_dir=tb_logname)
    else:
        tb_writer = DummySummaryWriter()

    kwargs = {'num_workers': conf.num_workers, 'pin_memory': conf.pin_memory} if conf.device == 'cuda' else {}

    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # model
    device = torch.device(conf.device)

    if conf.task_type == 'image_embedding':
        model = get_model(conf, logger)
        find_unused_parameters = False
        if conf.backbone_freeze_epochs > 0:
            find_unused_parameters = True
    else:
        raise NotImplementedError(conf.task_type)

    model = model.to(device=device)

    if conf.torch_compile:
        model = torch.compile(model)
        logger.info("torch.compile!")

    if hvd is not None:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    else:
        model = DDP(model, device_ids=[rank],
                    find_unused_parameters=find_unused_parameters)

    # setup dataset
    trn_idx = dataset[dataset['fold'] != fold].index
    val_idx = dataset[dataset['fold'] == fold].index

    train_folds = dataset.loc[trn_idx].reset_index(drop=True)
    valid_folds = dataset.loc[val_idx].reset_index(drop=True)
    train_feats = feats[trn_idx, :]
    valid_feats = feats[val_idx, :]

    if conf.val_fold is not None:
        val_idx = dataset[dataset['fold'] == conf.val_fold].index
        valid_folds = dataset.loc[val_idx].reset_index(drop=True)
        valid_feats = feats[val_idx, :]
        logger.info(f"Use fold{conf.val_fold} for validation")

    if conf.debug:
        debug_num_train = 8192
        debug_num_val = 2048
        train_folds = train_folds.head(debug_num_train)
        train_feats = train_feats[:debug_num_train,:]
        valid_folds = valid_folds.head(debug_num_val)
        valid_feats = valid_feats[:debug_num_val,:]
        logger.info("debug")

    assert train_folds.shape[0] == train_feats.shape[0]
    assert valid_folds.shape[0] == valid_feats.shape[0]

    train_dataset = get_dataset(conf, 'train', train_folds, train_feats, vis_root=conf.base_dir)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=conf.train_bs, drop_last=True, sampler=train_sampler, **kwargs)

    valid_dataset = get_dataset(conf, 'valid', valid_folds, valid_feats, vis_root=conf.base_dir)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=world_size, rank=rank)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=conf.valid_bs, drop_last=False, sampler=valid_sampler, **kwargs)

    # update print_freq
    if conf.print_freq == 0:
        conf.print_freq = max(2, len(train_loader) // 10)

    # model parameters
    no_decay = ['bias',
                'layernorm', 'layer_norm', 'LayerNorm',
                'layer_norm.bias', 'layer_norm.weight',
                'layernorm.bias', 'layernorm.weight'
                'LayerNorm.bias', 'LayerNorm.weight',
                'position_embedding', 'positional_embedding',
                'ln_pre.weight', 'ln_pre.bias', 'ln_1.weight', 'ln_1.bias',
                'ln_2.weight', 'ln_2.bias', 'ln_post.weight', 'ln_post.bias',
                'act.weight', 'act.bias',
    ]
    if conf.final_norm_no_decay:
        no_decay.extend(['norm.weight', 'norm.bias'])

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'lr': float(conf.lr),
            'weight_decay': float(conf.weight_decay)},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'lr': float(conf.lr),
            'weight_decay': 0.0},
    ]

    optimizer = get_optimizer(conf, optimizer_grouped_parameters)


    if hvd is not None:
        backward_passes_per_step = 1
        optimizer = hvd.DistributedOptimizer(
            optimizer, named_parameters=model.named_parameters(),
            compression=hvd.Compression.none,
            backward_passes_per_step=backward_passes_per_step)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    scheduler = get_scheduler(conf, optimizer, train_loader, logger)

    global_step = 0

    cossim = 0.
    loss = 10000.
    best_loss = 10000.
    best_cossim = 0.
    best_cossim_list = []
    full_prompts = []
    best_prompts = []
    full_feats = None
    best_feats = None

    current_lr = scheduler.get_last_lr()
    logger.info(f"lr: {current_lr}")
    tb_writer.add_scalar('Other/LearningRate', current_lr[0], global_step)

    scaler = GradScaler(enabled=conf.amp)

    for epoch in range(conf.epochs):

        start_time = time.time()

        # train
        if conf.train:

            loss, global_step = train_fn_emb_ddp(hvd, rank, local_rank, world_size, conf, epoch, global_step, train_loader,
                                                    model, scaler, optimizer, scheduler, device, train_sampler,
                                                    logger, tb_writer, fold, OUTPUT_DIR)

        if conf.skip_val:
            pass
        else:
            cossim, cossim_list, full_feats = valid_fn_emb_ddp(hvd, rank, local_rank, world_size, conf, epoch, global_step, valid_loader,
                                                model, valid_feats, device, logger, tb_writer)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(loss)
        elif isinstance(scheduler, CosineLRScheduler):
            if conf.cosine_interval_type == 'epoch':
                scheduler.step(epoch+1)

        current_lr = scheduler.get_last_lr()
        logger.info(f"lr: {current_lr}")
        tb_writer.add_scalar('Other/LearningRate', current_lr[0], global_step)

        if hvd is not None:
            target = model
        else:
            target = model.module

        ### model saving ###
        if conf.train and (loss < best_loss) and rank == 0:
            best_loss = loss
            # logger.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f}')
            # save_filename = f'{OUTPUT_DIR}/{conf.prefix}_fold{fold}_best_loss.pth'
            # save_checkpoint(conf, target, save_filename)

        if conf.train and (cossim > best_cossim) and rank == 0:
            best_cossim = cossim
            best_cossim_list = cossim_list
            best_prompts = full_prompts
            best_feats = full_feats
            logger.info(f'Epoch {epoch+1} - Save Best cossim: {best_cossim:.4f}')
            save_filename = f'{OUTPUT_DIR}/{conf.prefix}_fold{fold}_best_cossim.pth'
            save_checkpoint(conf, target, save_filename)

        if conf.extract_feat:
            best_feats = full_feats

        ### final epoch
        if conf.train and ((epoch+1) == conf.epochs) and rank == 0:
            save_filename = f'{OUTPUT_DIR}/{conf.prefix}_fold{fold}_epoch{epoch+1}.pth'
            save_checkpoint(conf, target, save_filename)

        elapsed = time.time() - start_time

        if conf.train:
            logger.info(f'Epoch {epoch+1} - Loss: {loss:.4f} Best: {best_loss:.4f} cossim: {cossim:.4f} Best: {best_cossim:.4f} time: {elapsed:.0f}s')
            logger.info(f"log: {log_path}")
        else:
            best_loss = None
            best_cossim = cossim
            best_cossim_list = cossim_list
            best_prompts = full_prompts
            best_feats = full_feats

        total_epochs -= 1
        full_elapsed = total_epochs * int(elapsed)
        time_delta = datetime.timedelta(seconds=full_elapsed)
        now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        end_time = now + time_delta
        logger.info(f"Expected remaining seconds: {full_elapsed} sec")
        logger.info(f"Expected remaining time: {time_delta}")
        logger.info(f"Expected end time: {end_time}")

    return best_loss, best_cossim, best_cossim_list, best_prompts, best_feats


if __name__ == '__main__':
    args = parse_args()
    conf = load_config(args.conf)

    for (key, value) in args._get_kwargs():
        if key in ['ckpt_path', 'amp_dtype',
                   'num_workers', 'valid_bs',
                   'save_feats', 'dataset_name',
                   'val_fold',
                   'test', 'seed', 'backend',
                   ]:
            if value is not None:
                setattr(conf, key, value)

    seed_everything(conf.seed)

    if args.debug:
        conf.debug = True

    if args.skip_val:
        conf.skip_val = True

    if args.save_feats or conf.save_feats:
        conf.save_feats = True

    if args.extract_feat or conf.extract_feat:
        conf.extract_feat = True
        conf.save_feats = True
        conf.train = False
        conf.epochs = 1

    if args.test or args.dataset_name:
        conf.train = False
        conf.skip_val = False
        conf.epochs = 1

    if args.backend == 'horovod':
        import horovod.torch as hvd
        hvd.init()
        rank = hvd.rank()
        local_rank = hvd.local_rank()
        world_size = hvd.size()
    else:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        hvd = None

    torch.manual_seed(conf.seed)

    torch.cuda.set_device(local_rank)
    torch.cuda.manual_seed(conf.seed)
    cudnn.benchmark = True

    formatter = logging.Formatter('%(message)s')
    logzero.formatter(formatter)

    if not os.path.exists(conf.log_dir):
        os.makedirs(conf.log_dir, exist_ok=True)

    save_dir = os.path.join(conf.log_dir, conf.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    log_basename = f"{conf.prefix}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    log_filename = f"{log_basename}.log"
    log_path = os.path.join(save_dir, log_filename)

    # メインのnode以外のloggerを停止
    if rank == 0:
        logzero.logfile(log_path)
    else:
        logzero.logfile('', disableStderrLogger=True)

    logger.info(conf)

    OUTPUT_DIR = f"{conf.model_dir}/{conf.save_dir}"

    logger.info(f"conf.ckpt_path: {conf.ckpt_path}")
    if (args.test or args.dataset_name) and (args.ckpt_path is None):
        # test時にはbest_lossのcheckpointを自動でloadする
        conf.ckpt_path = f'{OUTPUT_DIR}/{conf.prefix}_fold0_best_cossim.pth'
        logger.info(f"set ckpt_path: {conf.ckpt_path}")

    logger.info(f"dataset_name: {conf.dataset_name}")

    if rank == 0:
        os.makedirs(f"{OUTPUT_DIR}", exist_ok=True)
    logger.info(f"output_dir: {OUTPUT_DIR}")

    trn_fold = [int(elem) for elem in conf.train_fold.split(',')]

    total_epochs = conf.epochs * len(trn_fold)

    dataset, feats = get_dataset_and_feats(conf)

    for fold in range(conf.n_fold):
        if fold in trn_fold:
            best_loss, best_cossim, best_cossim_list, best_prompts, best_feats = \
                train_loop(conf, hvd, rank, local_rank, world_size,
                           dataset, feats, fold, logger, save_dir,
                           log_basename, log_path, total_epochs)

            num_datasets = len(conf.dataset_name.split(','))
            if num_datasets > 2:
                dataset_str = f"{num_datasets}datasets"
            else:
                dataset_str = conf.dataset_name.replace(',','_')

            if conf.val_fold is not None:
                dataset_str += f"_val_fold{conf.val_fold}"

            if conf.save_feats and (rank == 0) and (best_feats is not None):
                feats_path = f"{OUTPUT_DIR}/{conf.prefix}_{dataset_str}_fold{fold}_feats.npy"
                np.save(feats_path, best_feats)
                logger.info(f"feats saved: {feats_path}")

            total_epochs -= conf.epochs

    logger.info(f"log saved: {os.path.join(save_dir, log_filename)}")

