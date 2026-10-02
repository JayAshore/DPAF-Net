import argparse
import datetime
import os
import time
from functools import partial

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.DPAF import DPAF
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import seed_everything, show_config, worker_init_fn
from utils.utils_fit import fit_one_epoch


def count_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def profile_model_flops(model, input_shape, device):
    try:
        from thop import profile
    except Exception as exc:
        return None, f"thop import failed: {exc}"

    was_training = model.training
    model.eval()
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1], device=device)
    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops = 2 * macs
        return {
            "macs": float(macs),
            "flops": float(flops),
            "profile_params": float(params),
        }, None
    except Exception as exc:
        return None, f"FLOPs profile failed: {exc}"
    finally:
        model.train(was_training)


def benchmark_model_fps(model, input_shape, device, warmup=50, repeat=200):
    was_training = model.training
    model.eval()
    dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1], device=device)

    try:
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            start_time = time.perf_counter()
            for _ in range(repeat):
                _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start_time

        fps = repeat / elapsed if elapsed > 0 else 0.0
        latency_ms = elapsed * 1000.0 / repeat if repeat > 0 else 0.0
        return {
            "fps": fps,
            "latency_ms": latency_ms,
            "warmup": warmup,
            "repeat": repeat,
        }, None
    except Exception as exc:
        return None, f"FPS benchmark failed: {exc}"
    finally:
        model.train(was_training)


def append_model_profile_to_log(model, input_shape, device, run_log_path):
    total_params, trainable_params = count_model_params(model)
    flops_info, flops_error = profile_model_flops(model, input_shape, device)
    fps_info, fps_error = benchmark_model_fps(model, input_shape, device)

    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write("\nModel profile:\n")
        f.write(f"  Input shape: 1 x 3 x {input_shape[0]} x {input_shape[1]}\n")
        f.write(f"  Total params: {total_params:,} ({total_params / 1e6:.4f} M)\n")
        f.write(f"  Trainable params: {trainable_params:,} ({trainable_params / 1e6:.4f} M)\n")

        if flops_info is not None:
            f.write(f"  MACs: {flops_info['macs'] / 1e9:.6f} GMACs\n")
            f.write(f"  FLOPs: {flops_info['flops'] / 1e9:.6f} GFLOPs\n")
        else:
            f.write(f"  FLOPs: unavailable ({flops_error})\n")

        if fps_info is not None:
            f.write(f"  FPS: {fps_info['fps']:.2f}\n")
            f.write(f"  Latency: {fps_info['latency_ms']:.3f} ms/image\n")
            f.write(f"  FPS benchmark: batch_size=1, warmup={fps_info['warmup']}, repeat={fps_info['repeat']}\n")
        else:
            f.write(f"  FPS: unavailable ({fps_error})\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Training With Pytorch')

    # ------------------ Configuration ------------------#
    parser.add_argument('--cuda', action='store_true', default=True, help='Whether to use CUDA')
    parser.add_argument('--seed', type=int, default=11, help='Random seed')
    parser.add_argument('--num-classes', type=int, default=4, help='Number of classes')
    parser.add_argument('--input-shape', type=int, nargs=2, default=[224, 224],
                        help='Input image shape [H, W]')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='Whether to use pretrained model')
    parser.add_argument('--model-path', type=str, default='', help='Path to pretrained model weights')

    parser.add_argument('--init-epoch', type=int, default=0, help='Initial epoch')
    parser.add_argument('--unfreeze-epoch', type=int, default=200, help='Total training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')

    parser.add_argument('--init-lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--optimizer-type', type=str, default='adam', choices=['adam', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--weight-decay', type=float, default=0, help='Weight decay for optimizer')

    parser.add_argument('--lr-decay-type', type=str, default='cos', choices=['cos', 'step'],
                        help='Learning rate decay type')
    parser.add_argument('--save-period', type=int, default=10, help='Period to save model checkpoints')
    parser.add_argument('--save-dir', type=str, default='logs/',
                        help='Directory to save logs and model checkpoints')
    parser.add_argument('--voc-path', type=str, default=r'datasets/NEU-seg',
                        help='Path to VOCdevkit dataset')

    parser.add_argument('--num-workers', type=int, default=10, help='Number of workers for data loading')
    parser.add_argument('--dice-loss', action='store_true', default=True, help='Use Dice Dice loss')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ------------------ Seed ------------------#
    seed_everything(args.seed)
    cls_weights = np.ones(args.num_classes, np.float32)

    # ------------------ Device ------------------#
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # ------------------ Model ------------------#
    model = DPAF(num_classes=args.num_classes).train()
    if not args.pretrained:
        weights_init(model)
    if args.model_path != '':
        print(f"Loading weights from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)

    model_train = model.to(device)

    # ------------------ Loss History ------------------#
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(args.save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=args.input_shape)

    # ------------------ Run log (config + timing) ------------------#
    os.makedirs(log_dir, exist_ok=True)
    run_log_path = os.path.join(log_dir, "run_log.txt")
    run_start_ts = time.time()
    run_start_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Run start: {run_start_str}\n")
        f.write(f"PID: {os.getpid()}\n")
        f.write(f"Device: {device}\n")
        f.write("Args:\n")
        for k, v in sorted(vars(args).items()):
            f.write(f"  {k}: {v}\n")
        f.write("\n")

    # ------------------ Dataset ------------------#
    with open(os.path.join(args.voc_path, "ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.voc_path, "ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        num_classes=args.num_classes, backbone='DPAF', model_path=args.model_path,
        input_shape=args.input_shape, Init_Epoch=args.init_epoch, UnFreeze_Epoch=args.unfreeze_epoch,
        batch_size=args.batch_size, Init_lr=args.init_lr, Min_lr=args.min_lr, optimizer_type=args.optimizer_type,
        momentum=args.momentum, lr_decay_type=args.lr_decay_type, save_period=args.save_period,
        save_dir=args.save_dir, num_workers=args.num_workers, num_train=num_train, num_val=num_val
    )

    train_dataset = UnetDataset(train_lines, args.input_shape, args.num_classes, True, args.voc_path)
    val_dataset = UnetDataset(val_lines, args.input_shape, args.num_classes, False, args.voc_path)

    gen = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate,
                     worker_init_fn=partial(worker_init_fn, rank=0, seed=args.seed))
    gen_val = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate,
                         worker_init_fn=partial(worker_init_fn, rank=0, seed=args.seed))

    # ------------------ Optimizer & LR ------------------#
    optimizer = {
        'adam': optim.Adam(model.parameters(), args.init_lr, betas=(args.momentum, 0.999),
                           weight_decay=args.weight_decay),
        'sgd': optim.SGD(model.parameters(), args.init_lr, momentum=args.momentum, nesterov=True,
                         weight_decay=args.weight_decay)
    }[args.optimizer_type]

    lr_scheduler_func = get_lr_scheduler(args.lr_decay_type, args.init_lr, args.min_lr, args.unfreeze_epoch)

    # ------------------ Callbacks ------------------#
    eval_callback = EvalCallback(model, args.input_shape, args.num_classes, val_lines, args.voc_path,
                                 log_dir, args.cuda, eval_flag=True, period=5)

    # ------------------ Variables to track best MIOU ------------------#
    best_miou = 0.0
    best_miou_epoch = 0
    
    # ------------------ Training Loop ------------------#
    for epoch in range(args.init_epoch, args.unfreeze_epoch):
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        epoch_start_ts = time.time()
        fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                      num_train // args.batch_size, num_val // args.batch_size, gen, gen_val, args.unfreeze_epoch,
                      args.cuda, args.dice_loss, cls_weights, args.num_classes,
                      save_period=args.save_period,
                      save_dir=args.save_dir, log_dir=log_dir, local_rank=0,
                      eval_period=5)
        epoch_seconds = time.time() - epoch_start_ts
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write(f"Epoch {epoch + 1}/{args.unfreeze_epoch} time: {epoch_seconds:.2f}s\n")
        # ------------------ Update and print best MIOU ------------------#  
        miou_file = os.path.join(log_dir, "epoch_miou.txt")
        if os.path.exists(miou_file):
            with open(miou_file, 'r') as file:
                lines = file.readlines()
                if lines:
                    current_miou = float(lines[-1].strip())
                    if current_miou > best_miou:
                        best_miou = current_miou
                        best_miou_epoch = epoch + 1
                    print('✅  Best MIOU so far: %.4f (Epoch %d)' % (best_miou, best_miou_epoch))
                    
    # ------------------ Save Best MIOU to File ------------------#
    if os.path.exists(miou_file):
        with open(miou_file, 'a') as file:
            file.write(f"\n--- Training Complete ---\n")
            file.write(f"Best MIOU: {best_miou:.4f} achieved at Epoch {best_miou_epoch}\n")
            print(f'\n🎉 Training finished! Best MIOU {best_miou:.4f} saved to {miou_file}')

    run_end_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_seconds = time.time() - run_start_ts
    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write("\n")
        f.write(f"Run end: {run_end_str}\n")
        f.write(f"Total time: {run_seconds/60.0:.2f} min ({run_seconds:.2f} s)\n")
        f.write(f"Best mIoU: {best_miou:.4f} @ epoch {best_miou_epoch}\n")

    append_model_profile_to_log(model_train, args.input_shape, device, run_log_path)

    loss_history.writer.close()
