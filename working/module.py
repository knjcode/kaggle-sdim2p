import time
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, LambdaLR
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm

# from loss import CosineSimilarityLoss, get_upper_triangular_no_diag_flat
from optimizer import SAM
from util import AverageMeter, timeSince, save_checkpoint


# ddpでallgatherするためのhelper
def dist_all_gather(input_tensor, world_size, hvd):
    if hvd is not None:
        gathered_tensor = hvd.allgather(input_tensor)
    else:
        gathered_tensor = [
            torch.empty_like(input_tensor).cuda() for i in range(world_size)
        ]
        dist.all_gather(gathered_tensor, input_tensor.cuda())
        gathered_tensor = torch.cat(gathered_tensor)
    return gathered_tensor

# tensorじゃないものもallgatherしたい場合
def dist_all_gather_object(objects, world_size, hvd):
    if hvd is not None:
        outputs = hvd.allgather_object(objects)
    else:
        outputs = [
            None for i in range(world_size)
        ]
        dist.all_gather_object(outputs, objects)

    gathered_objects = []
    [gathered_objects.extend(elem) for elem in outputs]

    return gathered_objects


def train_fn_emb_ddp(hvd, rank, local_rank, world_size, conf, epoch, global_step, train_loader,
                     model, scaler, optimizer, scheduler, device,
                     train_sampler, logger, tb_writer, fold, OUTPUT_DIR):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    criterion = torch.nn.CosineEmbeddingLoss()

    model.train()

    if hvd is not None:
        target = model
    else:
        target = model.module

    if epoch < conf.backbone_freeze_epochs:
        for param in target.backbone.parameters():
            param.requires_grad = False
        target.backbone.eval()
        logger.info(f"Epoch: [{epoch+1}] Freeze backbone")
    elif epoch == conf.backbone_freeze_epochs:
        for param in target.backbone.parameters():
            param.requires_grad = True
        target.backbone.train()
        logger.info(f"Epoch: [{epoch+1}] Unfreeze backbone")

    train_sampler.set_epoch(epoch)

    if conf.amp_dtype == 'bfloat16':
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    start = end = time.time()

    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        global_step += 1

        # measure data loading time
        data_time.update(time.time() - end)
        image, emb, _ids = batch

        batch_size = image.size(0)

        image = image.to(device, non_blocking=True)
        emb = emb.to(device, non_blocking=True)
        cossim_target = torch.ones(1).to(device, non_blocking=True)

        with autocast(enabled=conf.amp, dtype=amp_dtype):
            feat = model(image)
            loss = criterion(feat, emb, cossim_target)

            # if conf.use_cossim_loss:
            #     target_cossim_list = get_upper_triangular_no_diag_flat(torch.mm(emb, emb.t()))
            #     cossim_loss = CosineSimilarityLoss(feat, target_cossim_list)
            #     loss = loss * conf.cosine_embedding_loss_weight + cossim_loss * conf.cosine_similarity_loss_weight


        losses.update(loss.item(), batch_size)

        if isinstance(optimizer, SAM):
            # SAM
            optimizer.step = optimizer.first_step
            with model.no_sync():
                scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            with autocast(enabled=conf.amp, dtype=amp_dtype):
                feat = model(image)
                loss = criterion(feat, emb, cossim_target)

            optimizer.step = optimizer.second_step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            scaler.scale(loss).backward()
            if hvd is not None:
                optimizer.synchronize()
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
                scaler.update()
                optimizer.synchronize()
                optimizer.zero_grad()
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        elif isinstance(scheduler, LambdaLR):
            scheduler.step()
        elif isinstance(scheduler, CosineLRScheduler):
            if conf.cosine_interval_type == 'step':
                scheduler.step(global_step+1)

        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % conf.print_freq == 0 or step == (len(train_loader)-1):
            logger.info(f'Epoch: [{epoch+1}][{step+1}/{len(train_loader)}] '
                f'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                f'Elapsed {timeSince(start, float(step+1)/len(train_loader)):s} '
                f'Loss: {losses.val:.4f}({losses.avg:.4f}) ')
            tb_writer.add_scalar('Loss/train', losses.avg, global_step)

            current_lr = scheduler.get_last_lr()
            tb_writer.add_scalar('Other/LearningRate', current_lr[0], global_step)

    if rank == 0 and conf.save_every_epoch:
        save_filename = f'{OUTPUT_DIR}/{conf.prefix}_fold{fold}_epoch{epoch+1}.pth'
        save_checkpoint(conf, target, save_filename)

    return losses.avg, global_step


def valid_fn_emb_ddp(hvd, rank, local_rank, world_size, conf, epoch, global_step, valid_loader,
                     model, valid_feats, device, logger, tb_writer):

    start = time.time()

    model.eval()

    feat_list = []
    # emb_list = []
    index_list = []

    if conf.amp_dtype == 'bfloat16':
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    disable = False if rank == 0 else True
    with torch.no_grad():
        for batch in tqdm(valid_loader, total=len(valid_loader), disable=disable):
            image, _emb, ids = batch
            image = image.to(device, non_blocking=True)

            with autocast(enabled=conf.amp, dtype=amp_dtype):
                feat = model(image)

            feat_list.append(feat.detach().cpu())
            # emb_list.append(emb)
            index_list.append(ids)


    feat = torch.cat(feat_list)

    pred_emb = dist_all_gather(feat, world_size, hvd)
    pred_emb = pred_emb.cpu().numpy()

    index_tensor = torch.cat(index_list).to(device, non_blocking=True)
    index_tensor = dist_all_gather(index_tensor, world_size, hvd)
    index_array = index_tensor.cpu().numpy()

    df_ids = pd.DataFrame(index_array, columns=['index'])
    df_ids = df_ids.drop_duplicates(subset=['index']).sort_values(by=['index'])

    df_pred_emb = pd.DataFrame(pred_emb)
    df_pred_emb = df_pred_emb.iloc[df_ids.index]
    full_pred_emb = torch.tensor(df_pred_emb.values)

    if conf.extract_feat:
        cossim = 0
        full_cossim_list = []
    else:
        cos = torch.nn.CosineSimilarity(dim=1)
        full_cossim_list = cos(full_pred_emb, torch.from_numpy(valid_feats)).numpy().tolist()

        cossim = np.mean(full_cossim_list)

        logger.info(f"Epoch {epoch+1} - cossim (Average): {cossim}")

        tb_writer.add_scalar('cossim', cossim, global_step)

    logger.info(f'Validation: Elapsed {timeSince(start, 1.):s} ')

    return cossim, full_cossim_list, full_pred_emb

