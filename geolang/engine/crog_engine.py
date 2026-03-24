import os
import time
import csv
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, trainMetricGPU, get_seg_image)
from utils.grasp_eval import (detect_grasps, calculate_iou, calculate_max_iou, calculate_jacquard_index, visualization)

def train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    ins_loss_meter = AverageMeter('Loss_ins', ':2.4f')
    qua_loss_metter = AverageMeter('Loss_qua', ':2.4f')
    sin_loss_metter = AverageMeter('Loss_sin', ':2.4f')
    cos_loss_metter = AverageMeter('Loss_cos', ':2.4f')
    wid_loss_metter = AverageMeter('Loss_wid', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time, data_time, lr, loss_meter, 
            ins_loss_meter,
            qua_loss_metter, sin_loss_metter, cos_loss_metter, wid_loss_metter, 
            iou_meter, pr_meter
        ],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    csv_path = None
    if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
        csv_path = os.path.join(args.output_dir, "train_metrics.csv")
        file_exists = os.path.exists(csv_path)
        if (not file_exists) or os.path.getsize(csv_path) == 0:
            with open(csv_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    "epoch",
                    "iter",
                    "total_iters",
                    "lr",
                    "loss",
                    "loss_avg",
                    "loss_ins",
                    "loss_ins_avg",
                    "loss_qua",
                    "loss_qua_avg",
                    "loss_sin",
                    "loss_sin_avg",
                    "loss_cos",
                    "loss_cos_avg",
                    "loss_wid",
                    "loss_wid_avg",
                    "iou",
                    "iou_avg",
                    "prec50",
                    "prec50_avg",
                    "batch_time",
                    "data_time",
                ])

    model.train()
    time.sleep(2)
    end = time.time()

    # size_list = [320, 352, 384, 416, 448, 480, 512]
    # idx = np.random.choice(len(size_list))
    # new_size = size_list[idx]

    for i, data in enumerate(train_loader):
        # image, target, text = data
        # ins_mask, grasp_quality_mask, grasp_sin_mask, grasp_cos_mask, grasp_width_mask = target
        
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        depth = data["depth"]
      
        
        
        data_time.update(time.time() - end)
        # data
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        ins_mask = ins_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.cuda(non_blocking=True).unsqueeze(1)
        
        # # multi-scale training
        # image = F.interpolate(image, size=(new_size, new_size), mode='bilinear')

        # forward
        # Depth (H,W) -> (1, H, W) -> (B, 1, H, W)  
        if depth is not None:
            depth = depth.cuda(non_blocking=True).unsqueeze(1)
        
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
            pred, target, loss, loss_dict = model(
                image,
                text,
                mask=ins_mask,
                grasp_qua_mask=grasp_qua_mask,
                grasp_sin_mask=grasp_sin_mask,
                grasp_cos_mask=grasp_cos_mask,
                grasp_wid_mask=grasp_wid_mask,
                depth=depth,
            )


        optimizer.zero_grad(set_to_none=True)

        if not torch.isfinite(loss):
            print("Loss is NaN/Inf")
            print(loss)

            for name, p in model.named_parameters():
                if not torch.isfinite(p).all():
                    print("Non-finite parameter before backward:", name)
                    raise RuntimeError(f"Bad parameter before backward: {name}")

            raise RuntimeError("Non-finite loss")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        for name, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                print("Non-finite gradient:", name)
                raise RuntimeError(f"Bad gradient: {name}")

        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        scaler.step(optimizer)

        for name, p in model.named_parameters():
            if not torch.isfinite(p).all():
                print("Non-finite parameter after step:", name)
                raise RuntimeError(f"Bad parameter after step: {name}")

        scaler.update()
        
        
        ins_mask_pred = pred[0]
        ins_mask_target = target[0]
        # DEBUG START
        # print("---- DEBUG METRICS INPUT ----")

        # print("pred shape:", ins_mask_pred.shape)
        # print("target shape:", ins_mask_target.shape)

        # print("pred min/max:", ins_mask_pred.min().item(), ins_mask_pred.max().item())
        # print("target min/max:", ins_mask_target.min().item(), ins_mask_target.max().item())

        # # If logits → apply sigmoid
        # prob = torch.sigmoid(ins_mask_pred)

        # print("prob min/max:", prob.min().item(), prob.max().item())
        # print("prob mean:", prob.mean().item())

        # pred_bin = (prob > 0.35).float()

        # print("pred positives:", pred_bin.sum().item())
        # print("gt positives:", ins_mask_target.sum().item())

        # print("-----------------------------")
        # DEBUG END

        # backward
        # optimizer.zero_grad()
        # if scaler is not None:
            
        #     if not torch.isfinite(loss):
        #         print("Loss is NaN/Inf")
        #         print(loss)
        #         exit()
        #     # print ("Loss is finite")
        #     scaler.scale(loss).backward()
        #     # print ("balle balle")

        #     scaler.unscale_(optimizer) # if i comment this then its the 1th AddMMBackward0 output, else Mmbackward0  in 0th output

        #     if args.max_norm:
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm) # max_norm

        #     scaler.step(optimizer)
        #     scaler.update()

        # else:
        #     loss.backward()
        #     # print ("Loss backward done")

        #     if args.max_norm:
        #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)

        #     optimizer.step()

                # metric
    
        iou, pr5 = trainMetricGPU(ins_mask_pred, ins_mask_target, 0.35, 0.5)
        # print(f"raw iou = {iou.item():.6f}, raw pr5 = {pr5.item():.6f}")

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss.detach())
            dist.all_reduce(iou)
            dist.all_reduce(pr5)
            loss = loss / dist.get_world_size()
            iou = iou / dist.get_world_size()
            pr5 = pr5 / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        ins_loss_meter.update(loss_dict["m_ins"], image.size(0))
        qua_loss_metter.update(loss_dict["m_qua"], image.size(0))
        sin_loss_metter.update(loss_dict["m_sin"], image.size(0))
        cos_loss_metter.update(loss_dict["m_cos"], image.size(0))
        wid_loss_metter.update(loss_dict["m_wid"], image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if csv_path is not None:
            with open(csv_path, "a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    epoch,
                    i + 1,
                    len(train_loader),
                    float(lr.val),
                    float(loss_meter.val),
                    float(loss_meter.avg),
                    float(ins_loss_meter.val),
                    float(ins_loss_meter.avg),
                    float(qua_loss_metter.val),
                    float(qua_loss_metter.avg),
                    float(sin_loss_metter.val),
                    float(sin_loss_metter.avg),
                    float(cos_loss_metter.val),
                    float(cos_loss_metter.avg),
                    float(wid_loss_metter.val),
                    float(wid_loss_metter.avg),
                    float(iou_meter.val),
                    float(iou_meter.avg),
                    float(pr_meter.val),
                    float(pr_meter.avg),
                    float(batch_time.val),
                    float(data_time.val),
                ])

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)
            use_wandb = getattr(args, "use_wandb", True)
            is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
            if use_wandb and is_rank0:
                global_step = (epoch - 1) * len(train_loader) + (i + 1)
                wandb.log(
                    {
                        "time/batch": float(batch_time.val),
                        "time/data": float(data_time.val),
                        "train/lr": float(lr.val),
                        "train/loss": float(loss_meter.val),
                        "train/loss_avg": float(loss_meter.avg),
                        "train/loss_ins": float(ins_loss_meter.val),
                        "train/loss_qua": float(qua_loss_metter.val),
                        "train/loss_sin": float(sin_loss_metter.val),
                        "train/loss_cos": float(cos_loss_metter.val),
                        "train/loss_wid": float(wid_loss_metter.val),
                        "train/iou": float(iou_meter.val),
                        "train/iou_avg": float(iou_meter.avg),
                        "train/prec@50": float(pr_meter.val),
                    },
                    step=global_step,
                )


@torch.no_grad()
def validate_with_grasp(val_loader, model, epoch, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderValue=0.)
        return inv_img

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    time.sleep(2)

    num_grasps = [1,5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]

    pbar = tqdm(val_loader)
    for data in pbar:
        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]
        
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        ins_mask = ins_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.cuda(non_blocking=True).unsqueeze(1)
        
        # inference & get predictions from model
        pred, target = model(
            image,
            text,
            mask=ins_mask,
            grasp_qua_mask=grasp_qua_mask,
            grasp_sin_mask=grasp_sin_mask,
            grasp_cos_mask=grasp_cos_mask,
            grasp_wid_mask=grasp_wid_mask,
        )
        
        # predictions
        ins_mask_preds = pred[0]
        grasp_qua_mask_preds = pred[1]
        grasp_sin_mask_preds = pred[2]
        grasp_cos_mask_preds = pred[3]
        grasp_wid_mask_preds = pred[4]
        
        # targets
        ins_mask_targets = target[0]
        grasp_qua_mask_targets = target[1]
        grasp_sin_mask_targets = target[2]
        grasp_cos_mask_targets = target[3]
        grasp_wid_mask_targets = target[4]
        
        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(ins_mask_preds)
        grasp_qua_mask_preds = torch.sigmoid(grasp_qua_mask_preds)
        grasp_wid_mask_preds = torch.sigmoid(grasp_wid_mask_preds)
        
        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_qua_mask_preds = F.interpolate(grasp_qua_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            
            grasp_sin_mask_preds = F.interpolate(grasp_sin_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            
            grasp_cos_mask_preds = F.interpolate(grasp_cos_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            
            grasp_wid_mask_preds = F.interpolate(grasp_wid_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
        
        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size
            
            ins_mask_pred = ins_mask_preds[idx].cpu().numpy()
            grasp_qua_mask_pred = grasp_qua_mask_preds[idx].squeeze().cpu().numpy()
            grasp_sin_mask_pred = grasp_sin_mask_preds[idx].squeeze().cpu().numpy()
            grasp_cos_mask_pred = grasp_cos_mask_preds[idx].squeeze().cpu().numpy()
            grasp_wid_mask_pred = grasp_wid_mask_preds[idx].squeeze().cpu().numpy()
            
            ins_mask_target = ins_mask_targets[idx].squeeze().cpu().numpy()
            grasp_target = grasp_targets[idx]
            grasp_qua_mask_target = grasp_qua_mask_targets[idx].squeeze().cpu().numpy()
            grasp_sin_mask_target = grasp_sin_mask_targets[idx].squeeze().cpu().numpy()
            grasp_cos_mask_target = grasp_cos_mask_targets[idx].squeeze().cpu().numpy()
            grasp_wid_mask_target = grasp_wid_mask_targets[idx].squeeze().cpu().numpy()
            
            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > 0.35)
            grasp_qua_mask_pred = inverse(grasp_qua_mask_pred, inv_mat, w, h)
            grasp_sin_mask_pred = inverse(grasp_sin_mask_pred, inv_mat, w, h)
            grasp_cos_mask_pred = inverse(grasp_cos_mask_pred, inv_mat, w, h)
            grasp_wid_mask_pred = inverse(grasp_wid_mask_pred, inv_mat, w, h)
            
            ins_mask_target = inverse(ins_mask_target, inv_mat, w, h)
            grasp_qua_mask_target = inverse(grasp_qua_mask_target, inv_mat, w, h)
            grasp_sin_mask_target = inverse(grasp_sin_mask_target, inv_mat, w, h)
            grasp_cos_mask_target = inverse(grasp_cos_mask_target, inv_mat, w, h)
            grasp_wid_mask_target = inverse(grasp_wid_mask_target, inv_mat, w, h)
            
            # Calculate IoU between predicted instance mask and gt
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)
            
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
            
            # Calculate grasp configurations
            for i in range(len(num_grasps)):
                num_g = num_grasps[i]
                grasp_preds, _ = detect_grasps(grasp_qua_mask_pred, grasp_sin_mask_pred, grasp_cos_mask_pred, grasp_wid_mask_pred, num_g)

                j_index = calculate_jacquard_index(grasp_preds, grasp_target)
                
                num_correct_grasps[i] += j_index
                num_total_grasps[i] += 1
    
    J_index = [0, 0]
    for i in range(len(num_grasps)):
        J_index[i] = num_correct_grasps[i]/num_total_grasps[i]
            
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(image.device)
    iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  J_index@1: {:.2f}  J_index@5: {:.2f}'.format(
        epoch, args.epochs, 100. * iou.item(), 100. * J_index[0], 100. * J_index[1])
    logger.info(head + temp)
    return iou.item(), prec, J_index


@torch.no_grad()
def validate_without_grasp(val_loader, model, epoch, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderValue=0.)
        return inv_img

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    time.sleep(2)

    num_grasps = [1,5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]

    pbar = tqdm(val_loader)
    for data in pbar:
        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]
        
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        ins_mask = ins_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.cuda(non_blocking=True).unsqueeze(1)
        
        # inference & get predictions from model
        pred, ins_mask_targets = model(
            image,
            text,
            mask=ins_mask,
            grasp_qua_mask=grasp_qua_mask,
            grasp_sin_mask=grasp_sin_mask,
            grasp_cos_mask=grasp_cos_mask,
            grasp_wid_mask=grasp_wid_mask,
        )

        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(pred)
        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
        
        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size
            
            ins_mask_pred = ins_mask_preds[idx].squeeze().cpu().numpy()
            ins_mask_target = ins_mask_targets[idx].squeeze().cpu().numpy()
            
            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > 0.35)
            
            ins_mask_target = inverse(ins_mask_target, inv_mat, w, h)
            
            # Calculate IoU between predicted instance mask and gt
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)
            
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
    
    J_index = [0, 0]
    
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(image.device)
    iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  J_index@1: {:.2f}  J_index@5: {:.2f}'.format(
        epoch, args.epochs, 100. * iou.item(), 100. * J_index[0], 100. * J_index[1])
    logger.info(head + temp)
    return iou.item(), prec, J_index



@torch.no_grad()
def inference_with_grasp(test_loader, model, args):
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderValue=0.)
        return inv_img

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    time.sleep(2)
    
    num_grasps = [1,5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]
    
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    for cnt, data in enumerate(tbar):
        
        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]
        sentences = data["sentence"]
        img_paths = data["img_path"]
        
        image = image.cuda(non_blocking=True)
        text = text.cuda(non_blocking=True)
        ins_mask = ins_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_qua_mask = grasp_qua_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_sin_mask = grasp_sin_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_cos_mask = grasp_cos_mask.cuda(non_blocking=True).unsqueeze(1)
        grasp_wid_mask = grasp_wid_mask.cuda(non_blocking=True).unsqueeze(1)
        
        # inference & get predictions from model
        pred, target = model(
            image,
            text,
            mask=ins_mask,
            grasp_qua_mask=grasp_qua_mask,
            grasp_sin_mask=grasp_sin_mask,
            grasp_cos_mask=grasp_cos_mask,
            grasp_wid_mask=grasp_wid_mask,
        )
        
        # predictions
        ins_mask_preds = pred[0]
        grasp_qua_mask_preds = pred[1]
        grasp_sin_mask_preds = pred[2]
        grasp_cos_mask_preds = pred[3]
        grasp_wid_mask_preds = pred[4]
        
        # targets
        ins_mask_targets = target[0]
        grasp_qua_mask_targets = target[1]
        grasp_sin_mask_targets = target[2]
        grasp_cos_mask_targets = target[3]
        grasp_wid_mask_targets = target[4]
        
        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(ins_mask_preds)
        grasp_qua_mask_preds = torch.sigmoid(grasp_qua_mask_preds)
        grasp_wid_mask_preds = torch.sigmoid(grasp_wid_mask_preds)
        
        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_qua_mask_preds = F.interpolate(grasp_qua_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            
            grasp_sin_mask_preds = F.interpolate(grasp_sin_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            
            grasp_cos_mask_preds = F.interpolate(grasp_cos_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            
            grasp_wid_mask_preds = F.interpolate(grasp_wid_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
        

        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size
            sent = sentences[idx]
            img_path = img_paths[idx]
            
            ins_mask_pred = ins_mask_preds[idx].cpu().numpy()
            grasp_qua_mask_pred = grasp_qua_mask_preds[idx].squeeze().cpu().numpy()
            grasp_sin_mask_pred = grasp_sin_mask_preds[idx].squeeze().cpu().numpy()
            grasp_cos_mask_pred = grasp_cos_mask_preds[idx].squeeze().cpu().numpy()
            grasp_wid_mask_pred = grasp_wid_mask_preds[idx].squeeze().cpu().numpy()
            
            ins_mask_target = ins_mask_targets[idx].squeeze().cpu().numpy()
            grasp_target = grasp_targets[idx]
            grasp_qua_mask_target = grasp_qua_mask_targets[idx].squeeze().cpu().numpy()
            grasp_sin_mask_target = grasp_sin_mask_targets[idx].squeeze().cpu().numpy()
            grasp_cos_mask_target = grasp_cos_mask_targets[idx].squeeze().cpu().numpy()
            grasp_wid_mask_target = grasp_wid_mask_targets[idx].squeeze().cpu().numpy()
            
            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > 0.35)
            grasp_qua_mask_pred = inverse(grasp_qua_mask_pred, inv_mat, w, h)
            grasp_sin_mask_pred = inverse(grasp_sin_mask_pred, inv_mat, w, h)
            grasp_cos_mask_pred = inverse(grasp_cos_mask_pred, inv_mat, w, h)
            grasp_wid_mask_pred = inverse(grasp_wid_mask_pred, inv_mat, w, h)
            
            ins_mask_target = inverse(ins_mask_target, inv_mat, w, h)
            grasp_qua_mask_target = inverse(grasp_qua_mask_target, inv_mat, w, h)
            grasp_sin_mask_target = inverse(grasp_sin_mask_target, inv_mat, w, h)
            grasp_cos_mask_target = inverse(grasp_cos_mask_target, inv_mat, w, h)
            grasp_wid_mask_target = inverse(grasp_wid_mask_target, inv_mat, w, h)
            
            # Calculate IoU between predicted instance mask and gt
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)
            
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
            
            # Calculate grasp configurations
            for i in range(len(num_grasps)):
                num_g = num_grasps[i]
                grasp_preds, grasp_ang_mask_pred = detect_grasps(grasp_qua_mask_pred, grasp_sin_mask_pred, grasp_cos_mask_pred, grasp_wid_mask_pred, num_g)

                j_index = calculate_jacquard_index(grasp_preds, grasp_target)
                
                num_correct_grasps[i] += j_index
                num_total_grasps[i] += 1
                
                # Visualization
                if args.visualize:
                    img_bgr = cv2.imread(img_path)
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    visualization(img, ins_mask_pred, (grasp_qua_mask_pred, grasp_ang_mask_pred, grasp_wid_mask_pred), grasp_preds, sent, save_path=os.path.join("./results", args.exp_name, f"results_{cnt}_{num_g}_grasps.png"))
                
    J_index = [0, 0]
    for i in range(len(num_grasps)):
        J_index[i] = num_correct_grasps[i]/num_total_grasps[i]
            
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(image.device)
    
    # print(iou_list)
    # iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('IoU={:.2f}'.format(100.*iou.item()))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))
    logger.info("J@1: {:.2f}, J@5: {:.2f}".format(100. * J_index[0], 100. * J_index[1]))

    return iou.item(), prec, J_index

      