from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from transformers import (
    get_constant_schedule_with_warmup,
)
from timm.scheduler.cosine_lr import CosineLRScheduler

class ReduceLROnPlateauPatch(ReduceLROnPlateau):
    def get_last_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]

class CosineLRSchedulerPatch(CosineLRScheduler):
    def get_last_lr(self):
        return [ group['lr'] for group in self.optimizer.param_groups ]

# ====================================================
# scheduler
# ====================================================
def get_scheduler(conf, optimizer, train_loader, logger):
    min_lr = float(conf.min_lr)
    warmup_lr_init = float(conf.warmup_lr_init)
    factor = float(conf.factor)
    eps = float(conf.plateau_eps)

    n_iter_per_epoch = len(train_loader)
    if n_iter_per_epoch == 0:
        n_iter_per_epoch = 1

    if conf.scheduler=='ReduceLROnPlateau':
        scheduler = ReduceLROnPlateauPatch(optimizer, mode=conf.plateau_mode,
                                           factor=factor, patience=conf.patience,
                                           verbose=True, eps=eps)
        logger.info(f"use ReduceLROnPlateau Scheduler factor:{factor} patience:{conf.patience}")
    elif conf.scheduler=='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=conf.epochs * n_iter_per_epoch,
                                      eta_min=min_lr, last_epoch=-1)
        logger.info(f"use CosineAnnealingLR Scheduler epochs:{conf.epochs}")
    elif conf.scheduler=='WarmupLinear':
        scheduler = get_constant_schedule_with_warmup(optimizer,
                                                      num_warmup_steps=conf.warmup_epochs * n_iter_per_epoch)
    elif conf.scheduler=='cosine': # timm CosineLRScheduler
        if conf.cosine_interval_type == 'step':
            scheduler = CosineLRSchedulerPatch(optimizer,
                                               t_initial=conf.epochs * n_iter_per_epoch,
                                               warmup_t=conf.warmup_epochs * n_iter_per_epoch,
                                               warmup_lr_init=warmup_lr_init,
                                               lr_min=min_lr)
        elif conf.cosine_interval_type == 'epoch':
            scheduler = CosineLRSchedulerPatch(optimizer,
                                               t_initial=conf.epochs,
                                               warmup_t=conf.warmup_epochs,
                                               warmup_lr_init=warmup_lr_init,
                                               lr_min=min_lr)
        else:
            raise NotImplementedError('invalid cosine_interval_type')

    return scheduler

