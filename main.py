import argparse
import random
import copy

from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

from engine import *
from build_modules import *
from datasets.augmentations import train_trans, val_trans, strong_trans
from utils import get_rank, init_distributed_mode, resume_and_load, save_ckpt, selective_reinitialize


def get_args_parser(parser):
    # Model Settings
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--pos_encoding', default='sine', type=str)
    parser.add_argument('--num_classes', default=9, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--with_box_refine', action="store_true")
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--feedforward_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--eval_batch_size', default=1, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj', default=2e-5, type=float)
    parser.add_argument('--sgd', action="store_true")
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.5, type=float, help='gradient clipping max norm')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--epoch_lr_drop', default=40, type=int)
    # Loss coefficients
    parser.add_argument('--only_class_loss', action="store_true")  # default: False, when define in config: it is True
    parser.add_argument('--high_quality_matches', action="store_true")
    parser.add_argument('--coef_class', default=2.0, type=float)
    parser.add_argument('--coef_boxes', default=5.0, type=float)
    parser.add_argument('--coef_giou', default=2.0, type=float)
    parser.add_argument('--alpha_focal', default=0.25, type=float)
    parser.add_argument('--alpha_ema', default=0.999, type=float)
    # Dataset parameters
    parser.add_argument('--data_root', default='./data', type=str)
    parser.add_argument('--source_dataset', default='cityscapes', type=str)
    parser.add_argument('--target_dataset', default='foggy_cityscapes', type=str)
    # Retraining parameters
    parser.add_argument('--keep_modules', default=["backbone", "encoder"], type=str, nargs="+")  # "decoder"
    # Masking parameters
    parser.add_argument('--block_size', default=64, type=int)
    parser.add_argument('--masked_ratio', default=0.5, type=float)
    parser.add_argument('--coef_masked_img', default=1.0, type=float)
    # Teaching parameters
    parser.add_argument('--dynamic_update', action="store_true")
    parser.add_argument('--fix_update_iter', default=1, type=int)
    parser.add_argument('--max_update_iter', default=5, type=int)
    parser.add_argument('--use_pseudo_label_weights', action="store_true")
    parser.add_argument('--use_loss_student', action="store_true")
    # Dynamic threshold (DT) parameters
    parser.add_argument('--threshold', default=0.3, type=float)
    parser.add_argument('--alpha_dt', default=0.5, type=float)
    parser.add_argument('--gamma_dt', default=0.9, type=float)
    parser.add_argument('--max_dt', default=0.45, type=float)
    # mode settings
    parser.add_argument("--mode", default="single_domain", type=str,
                        help="'single_domain' for single domain training,"
                             "'teaching_standard' for teaching standard process,"
                             "'teaching_mask' for teaching with mask process,"
                             "'eval' for evaluation only.")
    # Other settings
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--random_seed', default=8008, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--print_freq', default=100, type=int)
    parser.add_argument('--flush', default=True, type=bool)
    parser.add_argument("--resume", default="", type=str)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_loss(epoch, prefix, total_loss, loss_dict):
    writer.add_scalar(prefix + '/total_loss', total_loss, epoch)
    for k, v in loss_dict.items():
        writer.add_scalar(prefix + '/' + k, v, epoch)


def write_ap50(epoch, prefix, m_ap, ap_per_class, idx_to_class):
    writer.add_scalar(prefix + '/mAP50', m_ap, epoch)
    for idx, num in zip(idx_to_class.keys(), ap_per_class):
        writer.add_scalar(prefix + '/AP50_%s' % (idx_to_class[idx]['name']), num, epoch)


def single_domain_training(model, device):
    # Record the start time
    start_time = time.time()
    # Build dataloaders
    train_loader = build_dataloader(args, args.source_dataset, 'source', 'train', train_trans)
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    idx_to_class = val_loader.dataset.coco.cats
    # Prepare model for optimization
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.gpu])
    criterion = build_criterion(args, device)
    optimizer = build_optimizer(args, model)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    # Record the best mAP
    ap50_best = -1.0
    for epoch in range(args.epoch):
        # Set the epoch for the sampler
        if args.distributed and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        # Train for one epoch
        loss_train = train_one_epoch_standard(
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            clip_max_norm=args.clip_max_norm,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # write_loss(epoch, 'single_domain', loss_train)
        lr_scheduler.step()
        # Evaluate
        ap50_per_class, loss_val = evaluate(
            model=model,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # Save the best checkpoint
        map50 = np.asarray([ap for ap in ap50_per_class if ap > -0.001]).mean().tolist()
        if map50 > ap50_best:
            ap50_best = map50
            save_ckpt(model, output_dir/'model_best.pth', args.distributed)
        if epoch == args.epoch - 1:
            save_ckpt(model, output_dir/'model_last.pth', args.distributed)
        # Write the evaluation results to tensorboard
        # write_ap50(epoch, 'single_domain', map50, ap50_per_class, idx_to_class)
    # Record the end time
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Single-domain training finished. Time cost: ' + total_time_str +
          ' . Best mAP50: ' + str(ap50_best), flush=args.flush)


# Teaching
def teaching(model_stu, device):
    start_time = time.time()
    # Build dataloaders
    target_loader = build_dataloader_teaching(args, args.target_dataset, 'target', 'train')
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    idx_to_class = val_loader.dataset.coco.cats
    # Build teacher model
    model_tch = build_teacher(args, model_stu, device)
    # Build init student model
    init_model_stu = build_teacher(args, model_stu, device)
    # print(init_model_stu.keys())
    # Prepare model for optimization
    if args.distributed:
        model_stu = DistributedDataParallel(model_stu, device_ids=[args.gpu], find_unused_parameters=False)
        model_tch = DistributedDataParallel(model_tch, device_ids=[args.gpu])
        init_model_stu = DistributedDataParallel(init_model_stu, device_ids=[args.gpu])
    # Build criterion, optimizer and lr_scheduler
    criterion = build_criterion(args, device)
    criterion_pseudo = build_criterion(args, device)
    criterion_pseudo_weak = build_criterion(args, device, only_class_loss=args.only_class_loss)
    optimizer = build_optimizer(args, model_stu)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.epoch_lr_drop)
    # Initialize thresholds
    thresholds = [args.threshold] * args.num_classes
    # Record the best mAP
    ap50_best = -1.0

    # Initialize buffers
    stu_buffer_cost = []
    stu_buffer_img = []
    stu_buffer_mask = []
    res_dict = {'stu_ori': [], 'stu_now': [], 'update_iter': []}

    # Initialize masking
    masking = Masking(block_size=args.block_size, masked_ratio=args.masked_ratio)

    for epoch in range(args.epoch):
        # Set the epoch for the sampler
        if args.distributed and hasattr(target_loader.sampler, 'set_epoch'):
            target_loader.sampler.set_epoch(epoch)
        if args.mode == "teaching_mask":
            loss_train, loss_target_dict = train_one_epoch_teaching_mask(
                student_model=model_stu,
                teacher_model=model_tch,
                init_student_model=init_model_stu,
                criterion_pseudo=criterion_pseudo,
                criterion_pseudo_weak=criterion_pseudo_weak,
                target_loader=target_loader,
                optimizer=optimizer,
                thresholds=thresholds,
                coef_masked_img=args.coef_masked_img,
                alpha_ema=args.alpha_ema,
                device=device,
                epoch=epoch,
                keep_modules=args.keep_modules,
                clip_max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                masking=masking,
                flush=args.flush,
                fix_update_iter=args.fix_update_iter,
                max_update_iter=args.max_update_iter,
                dynamic_update=args.dynamic_update,
                stu_buffer_cost=stu_buffer_cost,
                stu_buffer_img=stu_buffer_img,
                stu_buffer_mask=stu_buffer_mask,
                res_dict=res_dict,
                use_pseudo_label_weights=args.use_pseudo_label_weights,
                use_loss_student=args.use_loss_student
            )
        elif args.mode == "teaching_standard":
            loss_train, loss_target_dict = train_one_epoch_teaching_standard(
                student_model=model_stu,
                teacher_model=model_tch,
                criterion_pseudo=criterion_pseudo,
                target_loader=target_loader,
                optimizer=optimizer,
                thresholds=thresholds,
                alpha_ema=args.alpha_ema,
                device=device,
                epoch=epoch,
                clip_max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                flush=args.flush,
                fix_update_iter=args.fix_update_iter,
            )
        else:
            raise ValueError('Invalid mode: ' + args.mode)

        # Renew thresholds
        # thresholds = criterion.dynamic_threshold(thresholds)
        # criterion.clear_positive_logits()

        # Write the losses to tensorboard
        write_loss(epoch, 'teaching_target', loss_train, loss_target_dict)
        lr_scheduler.step()

        # Evaluate teacher and student model
        ap50_per_class_teacher, loss_val_teacher = evaluate(
            model=model_tch,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        ap50_per_class_student, loss_val_student = evaluate(
            model=model_stu,
            criterion=criterion,
            data_loader_val=val_loader,
            device=device,
            print_freq=args.print_freq,
            flush=args.flush
        )
        # Save the best checkpoint
        map50_tch = np.asarray([ap for ap in ap50_per_class_teacher if ap > -0.001]).mean().tolist()
        map50_stu = np.asarray([ap for ap in ap50_per_class_student if ap > -0.001]).mean().tolist()
        write_ap50(epoch, 'teaching_teacher', map50_tch, ap50_per_class_teacher, idx_to_class)
        write_ap50(epoch, 'teaching_student', map50_stu, ap50_per_class_student, idx_to_class)
        # if max(map50_tch, map50_stu) > ap50_best:
        #     ap50_best = max(map50_tch, map50_stu)
        #     save_ckpt(model_tch if map50_tch > map50_stu else model_stu, output_dir/'model_best.pth', args.distributed)
        if map50_tch > ap50_best:
            ap50_best = map50_tch
            save_ckpt(model_tch, output_dir/'model_best.pth', args.distributed)
        if epoch == args.epoch - 1:
            save_ckpt(model_tch, output_dir/'model_last_tch.pth', args.distributed)
            save_ckpt(model_stu, output_dir/'model_last_stu.pth', args.distributed)
        # if (epoch+1) % 5 == 0:
        #     save_ckpt(model_tch, output_dir/f'tch_epoch{epoch:02}.pth', args.distributed)
        #     save_ckpt(model_stu, output_dir/f'stu_epoch{epoch:02}.pth', args.distributed)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching finished. Time cost: ' + total_time_str + ' . Best mAP50: ' + str(ap50_best), flush=args.flush)


# Evaluate only
def eval_only(model, device):
    if args.distributed:
        Warning('Evaluation with distributed mode may cause error in output result labels.')
    criterion = build_criterion(args, device)
    # Eval source or target dataset
    val_loader = build_dataloader(args, args.target_dataset, 'target', 'val', val_trans)
    ap50_per_class, epoch_loss_val, coco_data = evaluate(
        model=model,
        criterion=criterion,
        data_loader_val=val_loader,
        output_result_labels=True,
        device=device,
        print_freq=args.print_freq,
        flush=args.flush
    )
    print('Evaluation finished. mAPs: ' + str(ap50_per_class) + '. Evaluation loss: ' + str(epoch_loss_val))
    output_file = output_dir/'evaluation_result_labels.json'
    print("Writing evaluation result labels to " + str(output_file))
    with open(output_file, 'w', encoding='utf-8') as fp:
        json.dump(coco_data, fp)


def main():
    # Initialize distributed mode
    init_distributed_mode(args)
    # Set random seed
    if args.random_seed is None:
        args.random_seed = random.randint(1, 10000)
    set_random_seed(args.random_seed + get_rank())
    # Print args
    print('-------------------------------------', flush=args.flush)
    print('Logs will be written to ' + str(logs_dir))
    print('Checkpoints will be saved to ' + str(output_dir))
    print('-------------------------------------', flush=args.flush)
    for key, value in args.__dict__.items():
        print(key, value, flush=args.flush)
    # Build model
    device = torch.device(args.device)
    model = build_model(args, device)
    if args.resume != "":
        model = resume_and_load(model, args.resume, device)
    # Training or evaluation
    print('-------------------------------------', flush=args.flush)
    if args.mode == "single_domain":
        single_domain_training(model, device)
    elif args.mode == "teaching_standard" or args.mode == "teaching_mask":
        teaching(model, device)
    elif args.mode == "eval":
        eval_only(model, device)
    else:
        raise ValueError('Invalid mode: ' + args.mode)


if __name__ == '__main__':
    # Parse arguments
    parser_main = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    get_args_parser(parser_main)
    args = parser_main.parse_args()
    # Set output directory
    output_dir = Path(args.output_dir)
    logs_dir = output_dir/'data_logs'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(logs_dir))
    # Call main function
    main()