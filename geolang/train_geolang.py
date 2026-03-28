import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
from functools import partial
from collections import OrderedDict

os.environ.setdefault("WANDB_MODE", "offline")


import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR

import utils.config as config
import wandb
from utils.dataset import OCIDVLGDataset
from engine.crog_engine import train_with_grasp, validate_with_grasp, validate_without_grasp
from model import build_geolang
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Pytorch Referring Expression Segmentation')
    parser.add_argument('--config',
                        default='path to xxx.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    torch.multiprocessing.set_start_method('spawn', force=True)
    
    args = get_parser()
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)

    env_world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    args.launched_with_env = env_world_size > 1

    if args.launched_with_env:
        args.world_size = env_world_size
        args.rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))

        visible_gpus = torch.cuda.device_count()
        if visible_gpus <= 0:
            raise RuntimeError("No CUDA GPUs visible for distributed launch.")

        args.ngpus_per_node = visible_gpus
        if visible_gpus == 1:
            local_rank = 0
        else:
            local_rank = local_rank % visible_gpus

        main_worker(local_rank, args)
        return

    args.ngpus_per_node = torch.cuda.device_count()
    if args.ngpus_per_node <= 0:
        raise RuntimeError("No CUDA GPUs visible.")
    args.world_size = args.ngpus_per_node * args.world_size
    # mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, ), join=True)

    if args.world_size <= 1:
        main_worker(0, args)
        return
    
    children = []
    for i in range(args.world_size):
        subproc = mp.Process(target=main_worker, args=(i, args))
        children.append(subproc)
        subproc.start()

    for i in range(args.world_size):
        children[i].join()


def main_worker(gpu, args):
    args.output_dir = os.path.join(args.output_folder, args.exp_name)
    use_ddp = args.world_size > 1

    # local rank & global rank
    args.gpu = gpu
    if not getattr(args, "launched_with_env", False):
        args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.gpu,
                 filename="train.log",
                 mode="a")

    # dist init
    if use_ddp:
        init_method = "env://" if getattr(args, "launched_with_env", False) else args.dist_url
        dist.init_process_group(backend=args.dist_backend,
                                init_method=init_method,
                                world_size=args.world_size,
                                rank=args.rank)

    logger.info(
        f"DDP launch: use_ddp={use_ddp}, rank={args.rank}, "
        f"world_size={args.world_size}, local_gpu={args.gpu}, "
        f"launched_with_env={getattr(args, 'launched_with_env', False)}"
    )

    # wandb (rank-0 only)
    use_wandb = getattr(args, "use_wandb", True)
    wandb_mode = getattr(args, "wandb_mode", os.environ.get("WANDB_MODE", "offline"))
    wandb_project = getattr(args, "wandb_project", "CROG")
    if args.rank == 0 and use_wandb:
        wandb.init(
            job_type="training",
            mode=wandb_mode,
            config=dict(args),
            project=wandb_project,
            name=args.exp_name,
            tags=[str(args.dataset), str(args.version)],
        )
    if use_ddp:
        dist.barrier()

    # build model
    model, param_list = build_geolang(args)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # convert all BatchNorm layers to SyncBatchNorm for distributed training. This ensures that the batch statistics are synchronized across all processes, which can lead to more stable training and better performance when using multiple GPUs. However, it may introduce some overhead due to the synchronization step, so it's typically used when training with a large batch size across multiple GPUs.
    # logger.info(model)
    # logger.info(args)
    
    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            milestones=args.milestones,
                            gamma=args.lr_decay)
    # scaler = torch.cuda.amp.GradScaler(init_scale=2**10) # init_scale is the initial scale factor for the gradients. It helps to prevent underflow in the early stages of training when the gradients might be very small. You can adjust this value based on your model and dataset. A common choice is 2^10 or 2^16, but you may need to experiment to find the best value for your specific case.
    # scaler =None
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2**2,
        growth_interval=20000,
        backoff_factor=0.5,
        growth_factor=2.0
    )

    detect_anomaly = bool(getattr(args, "detect_anomaly", False))
    torch.autograd.set_detect_anomaly(detect_anomaly)
    if detect_anomaly and args.rank == 0:
        logger.warning("Autograd anomaly detection is ENABLED (debug mode, slower training).")
    
    # # resume
    # best_IoU = 0.0
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         logger.info("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(
    #             args.resume, map_location=torch.device('cpu'))
    #         args.start_epoch = checkpoint['epoch']
    #         best_IoU = checkpoint["best_iou"]
    #         state_dict = checkpoint['state_dict']
    #         new_state_dict = OrderedDict()
    #         for k, v in state_dict.items():
    #             name = k[7:] # remove `module.`
    #             new_state_dict[name] = v
    #         # load params
    #         model.load_state_dict(new_state_dict)
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         scheduler.load_state_dict(checkpoint['scheduler'])
    #         logger.info("=> loaded checkpoint '{}' (epoch {})".format(
    #             args.resume, checkpoint['epoch']))
    #     else:
    #         raise ValueError(
    #             "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
    #             .format(args.resume))
    
    
    
    model = model.cuda()

    # Ensure parameter storage/layout is canonical contiguous before DDP buckets are created.
    # For 1x1 conv weights, ambiguous size-1 strides can still be considered contiguous;
    # forcing contiguous_format removes reducer stride-mismatch warnings.
    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.ndim == 4:
                parameter.data = parameter.data.detach().clone().to(memory_format=torch.contiguous_format)
            else:
                parameter.data = parameter.data.detach().clone().contiguous()

    if use_ddp:
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[args.gpu],
                                                    find_unused_parameters=False,
                                                    gradient_as_bucket_view=False)

        # This is needed when using gradient checkpointing (VMamba) with DDP
        model._set_static_graph()

    # build dataset
    if use_ddp:
        if getattr(args, "launched_with_env", False):
            per_process_divisor = max(1, args.world_size)
        else:
            per_process_divisor = max(1, args.ngpus_per_node)
    else:
        per_process_divisor = 1

    args.batch_size = max(1, int(args.batch_size / per_process_divisor))
    args.batch_size_val = max(1, int(args.batch_size_val / per_process_divisor))
    args.workers = int((args.workers + per_process_divisor - 1) / per_process_divisor)
    args.workers_val = int((args.workers_val + per_process_divisor - 1) / per_process_divisor)

        
    train_data = OCIDVLGDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            word_length=args.word_len,
                            split='train',
                            version=args.version)
    val_data = OCIDVLGDataset(root_dir=args.root_path,
                            input_size=args.input_size,
                            word_length=args.word_len,
                            split='val',
                            version=args.version)
    
    # get item of train_data for one sample
    
    if args.rank == 0:
        sample = train_data[0]
        print ("Sample keys:", sample.keys())
        print ("Image shape:", sample['img'].shape)
        print ("Mask shape:", sample['mask'].shape)
        print ("depth shape:", sample['depth'].shape)
        

    # build dataloader
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)
    if use_ddp:
        train_sampler = data.distributed.DistributedSampler(train_data,
                                                            shuffle=True)
        val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=(train_sampler is None),
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   worker_init_fn=init_fn,
                                   sampler=train_sampler,
                                   drop_last=True,
                                   collate_fn=OCIDVLGDataset.collate_fn)
    val_loader = data.DataLoader(val_data,
                                 batch_size=args.batch_size_val,
                                 shuffle=False,
                                 num_workers=args.workers_val,
                                 pin_memory=True,
                                 sampler=val_sampler,
                                 drop_last=False,
                                 collate_fn=OCIDVLGDataset.collate_fn)

    best_IoU = 0.0
    best_j_index = 0.0
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu}
            checkpoint = torch.load(
                args.resume, map_location=map_location)
            ckpt_state = checkpoint['state_dict']
            model_state = model.state_dict()

            filtered_state = OrderedDict()
            shape_mismatch = []
            unexpected = []

            for k, v in ckpt_state.items():
                if k not in model_state:
                    unexpected.append(k)
                    continue
                if model_state[k].shape != v.shape:
                    shape_mismatch.append(
                        f"{k}: ckpt={tuple(v.shape)} model={tuple(model_state[k].shape)}")
                    continue
                filtered_state[k] = v

            incompatible = model.load_state_dict(filtered_state, strict=False)

            can_resume_training_state = (
                len(shape_mismatch) == 0
                and len(unexpected) == 0
                and len(incompatible.missing_keys) == 0
                and len(incompatible.unexpected_keys) == 0
            )

            if can_resume_training_state:
                args.start_epoch = checkpoint.get('epoch', args.start_epoch)
                best_IoU = checkpoint.get("best_iou", best_IoU)
                best_j_index = checkpoint.get("best_j_index", best_j_index)
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint.get('epoch', -1)))
            else:
                logger.warning(
                    "=> partial model load from checkpoint due to incompatibilities; "
                    "optimizer/scheduler not resumed.")
                logger.warning(
                    f"=> skipped keys: shape_mismatch={len(shape_mismatch)}, "
                    f"checkpoint_only={len(unexpected)}, model_only={len(incompatible.missing_keys)}")
                if shape_mismatch:
                    logger.warning("=> first shape mismatches: {}".format(shape_mismatch[:5]))
                if unexpected:
                    logger.warning("=> first checkpoint-only keys: {}".format(unexpected[:5]))
                if incompatible.missing_keys:
                    logger.warning("=> first model-only keys: {}".format(incompatible.missing_keys[:5]))
            
            del checkpoint
            torch.cuda.empty_cache()
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    # start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        if train_sampler is not None:
            train_sampler.set_epoch(epoch_log)

        # train
        train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch_log,  args)
        # evaluation
        if args.use_grasp_masks:
            iou, prec_dict, j_index = validate_with_grasp(val_loader, model, epoch_log, args)
        else:
            iou, prec_dict, j_index = validate_without_grasp(val_loader, model, epoch_log, args)

        if args.rank == 0 and use_wandb:
            val_payload = {
                "epoch": epoch_log,
                "val/iou": float(iou),
                "val/j_index@1": float(j_index[0]),
                "val/j_index@5": float(j_index[1]),
            }
            for key, value in prec_dict.items():
                val_payload[f"val/{key}"] = float(value)
            wandb.log(val_payload, step=epoch_log)

        # save model
        if (not use_ddp) or dist.get_rank() == 0:
            lastname = os.path.join(args.output_dir, "last_model.pth")
            torch.save(
                {
                    'epoch': epoch_log,
                    'cur_iou': iou,
                    'best_iou': best_IoU,
                    'best_j_index': best_j_index,
                    'prec': prec_dict,
                    'j_index': j_index,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, lastname)
            if iou >= best_IoU:
                best_IoU = iou
                bestname = os.path.join(args.output_dir, "best_iou_model.pth")
                shutil.copyfile(lastname, bestname)
            
            if j_index[0] >= best_j_index:
                best_j_index = j_index[0]
                bestname = os.path.join(args.output_dir, "best_jindex_model.pth")
                shutil.copyfile(lastname, bestname)

        # update lr
        scheduler.step(epoch_log)
        torch.cuda.empty_cache()

    time.sleep(2)
    if args.rank == 0 and use_wandb:
        wandb.finish()

    logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))

    if use_ddp and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
    sys.exit(0)