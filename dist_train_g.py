import argparse
import datetime
import math
import os
import stat
from itertools import cycle

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as Data
from torch import nn
from tqdm import tqdm

from configs.transforms import transforms_dict
from datasets.folder import ImageFolderG, ImageFolderUnsupG
from utils.gan import get_model, load_weights, update


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_workers = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    to_save = False if args.do_not_save or args.local_rank != 0 else True
    saving_path = args.saving_root
    if to_save and not os.path.exists(saving_path):
        os.makedirs(saving_path)
        os.chmod(saving_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    sp = "" if args.g_weights == "" else "   "

    train_path = os.path.join(args.data_root, args.dataset, args.sub, "train_few_shot_tgt" if args.fs else "train_tgt")
    val_path = os.path.join(args.data_root, args.dataset, "val_tgt" if args.fs else "val_tgt")

    if args.local_rank == 0:
        print(f"device: {device}\n"
              f"Using {num_workers} dataloader workers every process\n\n"
              f"{sp} train path: {train_path}\n"
              f"{sp}   val path: {val_path}\n" +
              (f"g-weights path: {args.g_weights}\n" if args.g_weights != "" else "") +
              (f"{sp}saving path: {saving_path}\n\n" if to_save else "\n") +
              f"dataset: {args.dataset} (src{len(os.listdir(train_path.replace('train_few_shot_tgt', 'train_src')))}-tgt{len(os.listdir(train_path))})" +
              (f" | using unsup.\n" if args.unsup or not args.fs else "\n") +
              (f"remarks: {args.remarks}\n\n" if args.remarks is not None else "\n"))

    if args.fs:
        train_dataset = ImageFolderG(root=train_path, transform=transforms_dict[args.dataset]["train"])
    else:
        train_dataset = ImageFolderUnsupG(root=train_path, transform=transforms_dict[args.dataset]["train"])
    val_dataset = ImageFolderG(root=val_path, transform=transforms_dict[args.dataset]["val"])

    train_sampler = Data.distributed.DistributedSampler(dataset=train_dataset, shuffle=False)
    val_sampler = Data.distributed.DistributedSampler(dataset=val_dataset, shuffle=False)

    train_loader = Data.DataLoader(dataset=train_dataset,
                                   sampler=train_sampler,
                                   batch_size=args.batch_size,
                                   pin_memory=True,
                                   num_workers=num_workers,
                                   collate_fn=ImageFolderG.collate_fn)

    val_loader = Data.DataLoader(dataset=val_dataset,
                                 sampler=val_sampler,
                                 batch_size=args.batch_size,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=ImageFolderG.collate_fn)

    if args.unsup:
        train_path_unsup = os.path.join(args.data_root, args.dataset, args.sub, "train_tgt")
        train_dataset_unsup = ImageFolderUnsupG(root=train_path_unsup, transform=transforms_dict[args.dataset]["train"])
        train_sampler_unsup = Data.distributed.DistributedSampler(dataset=train_dataset_unsup, shuffle=False)
        train_loader_unsup = Data.DataLoader(dataset=train_dataset_unsup,
                                             sampler=train_sampler_unsup,
                                             batch_size=args.batch_size,
                                             pin_memory=True,
                                             num_workers=num_workers,
                                             collate_fn=ImageFolderG.collate_fn)
        train_loader_unsup = cycle(train_loader_unsup)

    generator = get_model(dataset=args.dataset, mode='G', device=device)

    load_weights(model=generator, weights=args.g_weights, device=device)

    if args.local_rank == 0:
        print("\n")

    generator = nn.parallel.DistributedDataParallel(generator, device_ids=[args.local_rank])

    criterion = nn.BCELoss()
    optim_g = optim.Adam(generator.parameters(), lr=args.lr)
    lr_lambda = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler_g = lr_scheduler.LambdaLR(optim_g, lr_lambda=lr_lambda)

    for epoch in range(1, args.epochs + 1):
        generator.train()
        total_g, correct_g = 0, 0

        train_loader_ = tqdm(train_loader) if args.local_rank == 0 else train_loader

        for i, (img_pairs, similarities) in enumerate(train_loader_):
            img_pairs = img_pairs.to(device)
            similarities = similarities.to(device)

            g_out = generator(img_pairs)  # [B, 1]

            g_loss = criterion(g_out, similarities)

            total_g += similarities.shape[0]
            correct_g += ((g_out.detach() + 0.5).int() == similarities.int()).sum().item()
            acc_g_train = correct_g / total_g

            update(generator, g_loss, optim_g)

            if args.local_rank == 0:
                train_loader_.desc = f"epoch [{epoch}/{args.epochs}]  " \
                                     f"loss G: {g_loss:.3f} | " \
                                     f"acc G: {acc_g_train:.3f} | " \
                                     f"lr: {optim_g.param_groups[0]['lr']:.2e}"

            if args.unsup:
                img_pairs, similarities = next(train_loader_unsup)
                img_pairs = img_pairs.to(device)
                similarities = similarities.to(device)
                g_out = generator(img_pairs)  # [B, 1]
                g_loss = criterion(g_out, similarities)
                update(generator, g_loss, optim_g)

        scheduler_g.step()

        if epoch % args.eval_interval == 0:
            with torch.no_grad():
                generator.eval()
                total, correct = 0, 0

                val_loader_ = tqdm(val_loader) if args.local_rank == 0 else val_loader

                for i, (img_pairs, similarities) in enumerate(val_loader_):
                    img_pairs = img_pairs.to(device)
                    similarities = similarities.to(device)

                    g_out = generator(img_pairs)  # [B, 1]
                    g_loss = criterion(g_out, similarities)

                    total_per_batch = torch.IntTensor([similarities.shape[0]]).to(device)
                    dist.all_reduce(total_per_batch, op=dist.ReduceOp.SUM)
                    total += total_per_batch.item()

                    correct_per_batch = ((g_out + 0.5).int() == similarities.int()).sum()
                    dist.all_reduce(correct_per_batch, op=dist.ReduceOp.SUM)
                    correct += correct_per_batch.item()

                    acc_g_tgt = correct / total

                    if args.local_rank == 0:
                        val_loader_.desc = f"valid  loss G: {g_loss:.3f} | acc: {acc_g_tgt:.3f}"

                if to_save and epoch % args.saving_interval == 0:
                    if args.fs:
                        pth = f"pre-G-{args.dataset}-{args.sub.split('-')[0]}-shot-{epoch}.pth"
                    else:
                        pth = f"pre-G-{args.dataset}-{epoch}.pth"
                    pth_path = os.path.join(saving_path, pth)
                    torch.save(generator.state_dict(), pth_path)
                    os.chmod(pth_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    print("model saved")

        if args.local_rank == 0:
            print(f"remarks: {args.remarks}\n\n" if args.remarks is not None else "\n")

    dist.destroy_process_group()


if __name__ == '__main__':
    date_time = datetime.datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--eval-interval', type=int, default=1)

    parser.add_argument('--unsup', action='store_true', help='whether to use the unsupervised mechanism')

    # datasets
    parser.add_argument('--data-root', type=str, default='./datasets/datasets')
    parser.add_argument('--dataset', type=str, default='mini-imagenet')
    parser.add_argument('--sub', type=str, default='src50-tgt50')
    parser.add_argument('--fs', action='store_true', help='whether to train few-shot target')

    # weights
    parser.add_argument('--g-weights', type=str, help='initial g_weights path')

    # save weights
    parser.add_argument('--do-not-save', action="store_true")
    parser.add_argument('--saving-root', type=str, default='./pre-trained')
    parser.add_argument('--date-time', type=str,
                        default=f'{date_time.month:02d}-{date_time.day:02d}_'
                                f'{date_time.hour:02d}-{date_time.minute:02d}-{date_time.second:02d}')
    parser.add_argument('--saving-interval', type=int, default=1)

    # remarks
    parser.add_argument('--remarks', type=str)

    args = parser.parse_args()

    train(args)
