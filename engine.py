import time
import datetime
import json

import copy

import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets.coco_style_dataset import DataPreFetcher
from datasets.coco_eval import CocoEvaluator

from models.criterion import post_process, get_pseudo_labels, get_topk_outputs, SetCriterion
from utils.distributed_utils import is_main_process
from utils.box_utils import box_cxcywh_to_xyxy, convert_to_xywh
from collections import defaultdict
from typing import List

from datasets.masking import Masking
from scipy.optimize import linear_sum_assignment
from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou
from utils import selective_reinitialize


def train_one_epoch_standard(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             data_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    """
    Train the standard detection model, using only labelled training set source.
    """
    start_time = time.time()
    model.train()
    criterion.train()
    fetcher = DataPreFetcher(data_loader, device=device)
    images, masks, annotations = fetcher.next()
    # Training statistics
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    epoch_loss_dict = defaultdict(float)
    for i in range(len(data_loader)):
        # Forward
        out = model(images, masks)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # Record loss
        epoch_loss += loss.detach()
        for k, v in loss_dict.items():
            epoch_loss_dict[k] += v.detach().cpu().item()
        # Data pre-fetch
        images, masks, annotations = fetcher.next()
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(len(data_loader)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of training statistic
    epoch_loss /= len(data_loader)
    for k, v in epoch_loss_dict.items():
        epoch_loss_dict[k] /= len(data_loader)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_loss_dict


def train_one_epoch_teaching_standard(student_model: torch.nn.Module,
                                      teacher_model: torch.nn.Module,
                                      criterion_pseudo: torch.nn.Module,
                                      target_loader: DataLoader,
                                      optimizer: torch.optim.Optimizer,
                                      thresholds: List[float],
                                      alpha_ema: float,
                                      device: torch.device,
                                      epoch: int,
                                      clip_max_norm: float = 0.0,
                                      print_freq: int = 20,
                                      flush: bool = True,
                                      fix_update_iter: int = 1):
    """
    Train the student model with the teacher model, using only unlabeled training set target .
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion_pseudo.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)

    for iter in range(total_iters):
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)

        # Target student forward
        target_student_out = student_model(target_student_images, target_masks)
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        loss = target_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        if iter % fix_update_iter == 0:
            with torch.no_grad():
                state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                for key, value in state_dict.items():
                    state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                teacher_model.load_state_dict(state_dict)

        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


def train_one_epoch_teaching_mask(student_model: torch.nn.Module,
                                  teacher_model: torch.nn.Module,
                                  init_student_model: torch.nn.Module,
                                  criterion_pseudo: torch.nn.Module,
                                  criterion_pseudo_weak: torch.nn.Module,
                                  target_loader: DataLoader,
                                  optimizer: torch.optim.Optimizer,
                                  thresholds: List[float],
                                  coef_masked_img: float,
                                  alpha_ema: float,
                                  device: torch.device,
                                  epoch: int,
                                  keep_modules: List[str],
                                  clip_max_norm: float = 0.0,
                                  print_freq: int = 20,
                                  masking: Masking = None,
                                  flush: bool = True,
                                  fix_update_iter: int = 1,
                                  max_update_iter: int = 5,
                                  dynamic_update: bool = False,
                                  stu_buffer_cost: List[float] = None,
                                  stu_buffer_img: List[torch.Tensor] = None,
                                  stu_buffer_mask: List[torch.Tensor] = None,
                                  res_dict: dict = None,
                                  use_pseudo_label_weights: bool = False,
                                  use_loss_student: bool = False):
    """
    Train the student model with the teacher model, using only unlabeled training set target (plus masked target image)
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    init_student_model.train()
    criterion_pseudo.train()
    criterion_pseudo_weak.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)

    for iter in range(total_iters):
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)

        # Target student forward
        target_student_out = student_model(target_student_images, target_masks)
        # loss from pseudo labels of current teacher
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        # Masked target student forward
        masked_target_images = masking(target_student_images)
        masked_target_student_out = student_model(masked_target_images, target_masks)
        # loss from pseudo labels of current teacher
        masked_target_loss, masked_target_loss_dict = criterion_pseudo(masked_target_student_out, pseudo_labels)

        # Final loss
        loss = target_loss + coef_masked_img * masked_target_loss

        # Loss from pseudo labels of previous student (just testing, not used)
        # if use_loss_student:
        #     # Loss from pseudo labels of previous student
        #     with torch.no_grad():
        #         student_out = student_model(target_teacher_images, target_masks)
        #         pseudo_labels_student = get_pseudo_labels(student_out['logit_all'][-1], student_out['boxes_all'][-1],
        #                                                   thresholds)
        #     target_loss_student, target_loss_dict_student = criterion_pseudo_weak(target_student_out,
        #                                                                         pseudo_labels_student, use_pseudo_label_weights)
        #     masked_target_loss_student, masked_target_loss_dict_student = criterion_pseudo_weak(masked_target_student_out,
        #                                                                                       pseudo_labels_student, use_pseudo_label_weights)
        #
        #     # Final loss
        #     loss_student = target_loss_student + coef_masked_img * masked_target_loss_student
        #     loss += loss_student

        # Dynamic update EMA teacher : Create buffer cost and buffer image in student model
        if dynamic_update:
            with torch.no_grad():
                student_out = student_model(target_teacher_images, target_masks)
            # variance logit
            student_out_var = student_out['logit_all'].var(dim=0)
            var_total = student_out_var.mean().item()
            stu_buffer_cost.append(var_total)

            # Store batch data to buffer
            stu_buffer_img.append(target_teacher_images.clone().detach())
            stu_buffer_mask.append(target_masks.clone().detach())

            if len(stu_buffer_cost) == 1:
                with torch.no_grad():
                    init_student_model.load_state_dict(student_model.state_dict())

            if len(stu_buffer_cost) >= 1:
                with torch.no_grad():
                    init_student_out = init_student_model(target_teacher_images, target_masks)
                    pseudo_labels_init_student = get_pseudo_labels(init_student_out['logit_all'][-1], init_student_out['boxes_all'][-1],
                                                              thresholds)
                # Loss from pseudo labels of init student
                init_student_loss, init_student_loss_dict = criterion_pseudo_weak(target_student_out,
                                                                                    pseudo_labels_init_student, use_pseudo_label_weights)
                masked_init_student_loss, masked_init_student_loss_dict = criterion_pseudo_weak(masked_target_student_out,
                                                                                                  pseudo_labels_init_student, use_pseudo_label_weights)
                loss_init_student = init_student_loss + coef_masked_img * masked_init_student_loss
                loss += loss_init_student

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        # Dynamic update EMA teacher : Update weight of teacher model
        if dynamic_update:
            if len(stu_buffer_cost) < max_update_iter:
                all_score = eval_stu(student_model, stu_buffer_img, stu_buffer_mask)
                compare_score = np.array(all_score) - np.array(stu_buffer_cost)
                # print(len(stu_buffer_cost), len(all_score), np.mean(compare_score<0))
                if np.mean(compare_score < 0) >= 0.5:
                    res_dict['stu_ori'].append(stu_buffer_cost)
                    res_dict['stu_now'].append(all_score)
                    res_dict['update_iter'].append(len(stu_buffer_cost))

                    df = pd.DataFrame(res_dict)
                    df.to_csv('dynamic_update.csv')

                    with torch.no_grad():
                        state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                        for key, value in state_dict.items():
                            state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                        teacher_model.load_state_dict(state_dict)

                    # Clear buffer
                    stu_buffer_cost = []
                    stu_buffer_img = []
                    stu_buffer_mask = []
            else:
                # print(len(stu_buffer_cost), 'Load previous student model weight')
                with torch.no_grad():
                    student_model = selective_reinitialize(student_model, init_student_model.state_dict(), keep_modules)

                # Clear buffer
                stu_buffer_cost = []
                stu_buffer_img = []
                stu_buffer_mask = []
        else:
            # EMA update teacher after fix iteration
            if iter % fix_update_iter == 0:
                with torch.no_grad():
                    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                    for key, value in state_dict.items():
                        state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                    teacher_model.load_state_dict(state_dict)


        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader_val: DataLoader,
             device: torch.device,
             print_freq: int,
             output_result_labels: bool = False,
             flush: bool = False):
    start_time = time.time()
    model.eval()
    criterion.eval()
    if hasattr(data_loader_val.dataset, 'coco') or hasattr(data_loader_val.dataset, 'anno_file'):
        evaluator = CocoEvaluator(data_loader_val.dataset.coco)
        coco_data = json.load(open(data_loader_val.dataset.anno_file, 'r'))
        # dataset_annotations = [[] for _ in range(len(coco_data['images']))]
        dataset_annotations = defaultdict(list)
    else:
        raise ValueError('Unsupported dataset type.')
    epoch_loss = 0.0
    for i, (images, masks, annotations) in enumerate(data_loader_val):
        # To CUDA
        images = images.to(device)
        masks = masks.to(device)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # Forward
        out = model(images, masks)
        logit_all, boxes_all = out['logit_all'], out['boxes_all']
        # Get pseudo labels
        if output_result_labels:
            results = get_pseudo_labels(logit_all[-1], boxes_all[-1], [0.4 for _ in range(9)])
            for anno, res in zip(annotations, results):
                image_id = anno['image_id'].item()
                orig_image_size = anno['orig_size']
                img_h, img_w = orig_image_size.unbind(0)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h])
                converted_boxes = convert_to_xywh(box_cxcywh_to_xyxy(res['boxes'] * scale_fct))
                converted_boxes = converted_boxes.detach().cpu().numpy().tolist()
                for label, box in zip(res['labels'].detach().cpu().numpy().tolist(), converted_boxes):
                    pseudo_anno = {
                        'id': 0,
                        'image_id': image_id,
                        'category_id': label,
                        'iscrowd': 0,
                        'area': box[-2] * box[-1],
                        'bbox': box
                    }
                    # dataset_annotations[image_id].append(pseudo_anno)
                    dataset_annotations[image_id].append(pseudo_anno)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        epoch_loss += loss
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Evaluation : [ ' + str(i + 1) + '/' + str(len(data_loader_val)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
        # mAP
        orig_image_sizes = torch.stack([anno['orig_size'] for anno in annotations], dim=0)
        results = post_process(logit_all[-1], boxes_all[-1], orig_image_sizes, 100)
        results = {anno['image_id'].item(): res for anno, res in zip(annotations, results)}
        evaluator.update(results)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    aps = evaluator.summarize()
    epoch_loss /= len(data_loader_val)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Evaluation finished. Time cost: ' + total_time_str, flush=flush)
    # Save results
    if output_result_labels:
        dataset_annotations_return = []
        id_cnt = 0
        # for image_anno in dataset_annotations:
        for image_anno in dataset_annotations.values():
            for box_anno in image_anno:
                box_anno['id'] = id_cnt
                id_cnt += 1
                dataset_annotations_return.append(box_anno)
        coco_data['annotations'] = dataset_annotations_return
        return aps, epoch_loss / len(data_loader_val), coco_data
    return aps, epoch_loss / len(data_loader_val)


def eval_stu(student_model: torch.nn.Module,
             stu_buffer_img: List[torch.Tensor],
             stu_buffer_mask: List[torch.Tensor]):
    """
    Evaluate student model with variance of logit
    """
    student_model.eval()
    all_score = []
    with torch.no_grad():
        for i in range(len(stu_buffer_img)):
            # student_out['logit_all']: [num_decoder_layers, batch size, num_queries, num_classes]
            student_out = student_model(stu_buffer_img[i], stu_buffer_mask[i])

            student_out_var = student_out['logit_all'].var(dim=0)
            var_total = student_out_var.mean().item()
            all_score.append(var_total)

    return all_score
