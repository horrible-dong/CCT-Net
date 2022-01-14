import argparse
import datetime
import math
import os
import stat
from itertools import cycle

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as Data
from tqdm import tqdm

from configs.transforms import transforms_dict
from datasets.folder import ImageFolderG, ImageFolderD, ImageFolderUnsupG
from utils.gan import get_model, load_weights, update, dist_evaluate


def train(args):
    pre, cond = args.sub.split("-")[:2]
    is_few_shot = True if cond == "shot" else False
    mode = f"{pre.replace('p', '%')}-shot" if is_few_shot else "zero-shot"

    batch_size, epochs = args.batch_size, args.epochs
    g_src_steps, g_fs_tgt_steps, g_steps, d_steps = args.g_src_steps, args.g_fs_tgt_steps, args.g_steps, args.d_steps
    if args.g_weights is None:
        args.g_weights = os.path.join(args.weights_root, f"pre-G-{args.dataset}.pth")
    sp = ""

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    to_save = False if args.do_not_save or args.local_rank != 0 else True
    saving_path = os.path.join(args.saving_root, args.date_time)
    if to_save and not os.path.exists(saving_path):
        os.makedirs(saving_path)
        os.chmod(saving_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    train_src_path = os.path.join(args.data_root, args.dataset, args.sub, "train_src")
    train_tgt_path = os.path.join(args.data_root, args.dataset, args.sub, "train_tgt")
    val_src_path = os.path.join(args.data_root, args.dataset, "val_src")
    val_tgt_path = os.path.join(args.data_root, args.dataset, "val_tgt")

    train_src_dataset_d = ImageFolderD(root=train_src_path, transform=transforms_dict[args.dataset]["train"])
    train_tgt_dataset_g = ImageFolderUnsupG(root=train_tgt_path, transform=transforms_dict[args.dataset]["train"])
    val_tgt_dataset_g = ImageFolderG(root=val_tgt_path, transform=transforms_dict[args.dataset]["val"])

    train_src_sampler_d = Data.distributed.DistributedSampler(dataset=train_src_dataset_d, shuffle=False)
    train_tgt_sampler_g = Data.distributed.DistributedSampler(dataset=train_tgt_dataset_g, shuffle=False)
    val_tgt_sampler_g = Data.distributed.DistributedSampler(dataset=val_tgt_dataset_g, shuffle=False)

    train_src_loader_d = Data.DataLoader(dataset=train_src_dataset_d,
                                         sampler=train_src_sampler_d,
                                         batch_size=batch_size,
                                         pin_memory=True,
                                         num_workers=num_workers,
                                         collate_fn=ImageFolderD.collate_fn)

    train_tgt_loader_g = Data.DataLoader(dataset=train_tgt_dataset_g,
                                         sampler=train_tgt_sampler_g,
                                         batch_size=batch_size,
                                         pin_memory=True,
                                         num_workers=num_workers,
                                         collate_fn=ImageFolderG.collate_fn)

    train_src_loader_d = cycle(train_src_loader_d)
    train_tgt_loader_g = cycle(train_tgt_loader_g)

    val_tgt_loader_g = Data.DataLoader(dataset=val_tgt_dataset_g,
                                       sampler=val_tgt_sampler_g,
                                       batch_size=batch_size,
                                       pin_memory=True,
                                       num_workers=num_workers,
                                       collate_fn=ImageFolderG.collate_fn)

    if args.g_src:
        train_src_dataset_g = ImageFolderG(root=train_src_path, transform=transforms_dict[args.dataset]["train"])
        train_src_sampler_g = Data.distributed.DistributedSampler(dataset=train_src_dataset_g, shuffle=False)
        train_src_loader_g = Data.DataLoader(dataset=train_src_dataset_g,
                                             sampler=train_src_sampler_g,
                                             batch_size=batch_size,
                                             pin_memory=True,
                                             num_workers=num_workers,
                                             collate_fn=ImageFolderG.collate_fn)
        train_src_loader_g = cycle(train_src_loader_g)
    else:
        g_src_steps = 0
        args.reducing = False

    if is_few_shot and args.g_fs_tgt_steps > 0:
        train_fs_tgt_path = os.path.join(args.data_root, args.dataset, args.sub, "train_few_shot_tgt")
        train_fs_tgt_dataset_g = ImageFolderG(root=train_fs_tgt_path, transform=transforms_dict[args.dataset]["train"])
        train_fs_tgt_sampler_g = Data.distributed.DistributedSampler(dataset=train_fs_tgt_dataset_g, shuffle=False)
        train_fs_tgt_loader_g = Data.DataLoader(dataset=train_fs_tgt_dataset_g,
                                                sampler=train_fs_tgt_sampler_g,
                                                batch_size=batch_size,
                                                pin_memory=True,
                                                num_workers=num_workers,
                                                collate_fn=ImageFolderG.collate_fn)
        train_fs_tgt_loader_g = cycle(train_fs_tgt_loader_g)
        sp = "   "
    else:
        g_fs_tgt_steps = 0

    if args.local_rank == 0:
        print(f"\n"
              f"device: {device}\n"
              f"Using {num_workers} dataloader workers every process\n\n"
              f"{sp}train src path: {train_src_path}\n"
              f"{sp}train tgt path: {train_tgt_path}\n" +
              (f"train fs tgt path: {train_fs_tgt_path}\n" if is_few_shot and args.g_fs_tgt_steps > 0 else "") +
              f"  {sp}val src path: {val_src_path}\n"
              f"  {sp}val tgt path: {val_tgt_path}\n" +
              (f"{sp}g-weights path: {args.g_weights}\n" if args.g_weights != "" else "") +
              (f"{sp}d-weights path: {args.d_weights}\n" if args.d_weights != "" else "") +
              (f"   {sp}saving path: {saving_path}\n\n" if to_save else "\n") +
              f"dataset: {args.dataset} (src{len(os.listdir(train_src_path))}-tgt{len(os.listdir(train_tgt_path))})"
              f" | mode: {mode} | batch size: {batch_size} | epochs: {epochs}"
              f" | G src steps: {g_src_steps}{' (reducing)' if args.reducing and g_src_steps > 0 else ''}" +
              (f" | G few-shot tgt steps: {g_fs_tgt_steps}" if is_few_shot else "") +
              f" | G steps: {g_steps} | D steps: {d_steps}" +
              (f" | no G weights\n" if args.g_weights == "" else "\n") +
              (f"remarks: {args.remarks}\n" if args.remarks is not None else "\n"))

    generator = get_model(dataset=args.dataset, mode='G', device=device)
    discriminator = get_model(dataset=args.dataset, mode='D', device=device)

    load_weights(model=generator, weights=args.g_weights, device=device)
    load_weights(model=discriminator, weights=args.d_weights, device=device)
    print()

    generator = nn.parallel.DistributedDataParallel(generator, device_ids=[args.local_rank])
    discriminator = nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.local_rank])

    criterion = nn.BCELoss()
    optim_d = optim.Adam(discriminator.parameters(), lr=args.lr_d)
    optim_g = optim.Adam(generator.parameters(), lr=args.lr_g)
    lr_lambda = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler_d = lr_scheduler.LambdaLR(optim_d, lr_lambda=lr_lambda)
    scheduler_g = lr_scheduler.LambdaLR(optim_g, lr_lambda=lr_lambda)

    steps_per_epoch = math.ceil(
        math.ceil(len(train_tgt_dataset_g) / dist.get_world_size() / batch_size) / (g_steps + d_steps - 1))

    steps_per_epoch = range(steps_per_epoch)
    best_acc_g_tgt = 0.

    dist_evaluate(generator, discriminator, val_tgt_loader_g, criterion, device, args.local_rank)

    dist.barrier()

    if args.local_rank == 0:
        print("\n")

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()

        if args.reducing:
            if epoch > 2:
                g_src_steps = 0
                args.reducing = False
                if args.local_rank == 0:
                    print(f"G src steps is now reduced to {g_src_steps}\n")

        total_g_tgt, correct_g_tgt, total_d_src, correct_d_src = 0, 0, 0, 0

        steps_per_epoch_ = tqdm(steps_per_epoch) if args.local_rank == 0 else steps_per_epoch

        for _ in steps_per_epoch_:
            # train G with src
            for _ in range(g_src_steps):
                src_pairs, src_similarities = next(train_src_loader_g)
                src_pairs = src_pairs.to(device)
                src_similarities = src_similarities.to(device)

                g_out = generator(src_pairs)  # [B, 1]
                g_loss = criterion(g_out, src_similarities)

                update(generator, g_loss, optim_g)

            # train G with few-shot tgt
            for _ in range(g_fs_tgt_steps):
                tgt_pairs, tgt_similarities = next(train_fs_tgt_loader_g)
                tgt_pairs = tgt_pairs.to(device)
                tgt_similarities = tgt_similarities.to(device)

                g_out = generator(tgt_pairs)  # [B, 1]
                g_loss = criterion(g_out, tgt_similarities)

                update(generator, g_loss, optim_g)

            # train D with src & tgt
            total_d_loss = 0
            for _ in range(d_steps):
                (src_pairs, src_similarities), src_labels = next(train_src_loader_d)
                src_pairs = src_pairs.to(device)
                src_similarities = src_similarities.to(device)
                src_labels = src_labels.to(device)

                tgt_pairs, tgt_similarities = next(train_tgt_loader_g)
                tgt_pairs = tgt_pairs.to(device)
                tgt_similarities = tgt_similarities.to(device)

                d_src_out = discriminator(src_pairs, condition=src_similarities)  # [B, 1]
                g_out = generator(tgt_pairs)  # [B, 1]
                d_tgt_out = discriminator(tgt_pairs, condition=g_out.detach())  # [B, 1]

                d_src_loss = criterion(d_src_out, src_labels)
                d_tgt_loss = criterion(d_tgt_out, torch.zeros_like(d_tgt_out))
                d_loss = (d_src_loss + d_tgt_loss) / 2
                total_d_loss += d_loss.item()

                update(discriminator, d_loss, optim_d)

            if args.local_rank == 0:
                # test G with tgt
                total_g_tgt += tgt_similarities.shape[0]
                correct_g_tgt += ((g_out.detach() + 0.5).int() == tgt_similarities.int()).sum().item()

                # test D with src
                total_d_src += src_labels.shape[0]
                correct_d_src += ((d_src_out.detach() + 0.5).int() == src_labels.int()).sum().item()

            # train G with tgt
            total_g_loss = 0
            for step in range(g_steps):
                d_tgt_out = discriminator(tgt_pairs, condition=g_out)  # [B, 1]

                g_loss = criterion(d_tgt_out, torch.ones_like(d_tgt_out))
                total_g_loss += g_loss.item()

                update(generator, g_loss, optim_g)

                if step < g_steps - 1:
                    tgt_pairs, tgt_similarities = next(train_tgt_loader_g)
                    g_out = generator(tgt_pairs)  # [B, 1]

            if args.local_rank == 0:
                avg_g_loss, avg_d_loss = total_g_loss / g_steps, total_d_loss / d_steps
                avg_g_tgt, avg_d_src = correct_g_tgt / total_g_tgt, correct_d_src / total_d_src
                steps_per_epoch_.desc = f"epoch [{epoch}/{epochs}]  " \
                                        f"loss G: {avg_g_loss:.3f} | " \
                                        f"loss D: {avg_d_loss:.3f} | " \
                                        f"acc G (t): {avg_g_tgt:.3f} | " \
                                        f"acc D (s): {avg_d_src:.3f} | " \
                                        f"lr G: {optim_g.param_groups[0]['lr']:.2e} | " \
                                        f"lr D: {optim_d.param_groups[0]['lr']:.2e}"

        scheduler_d.step()
        scheduler_g.step()

        if epoch % args.eval_interval == 0:
            acc_g_tgt = dist_evaluate(generator, discriminator, val_tgt_loader_g, criterion, device, args.local_rank)
            if acc_g_tgt > best_acc_g_tgt:
                best_acc_g_tgt = acc_g_tgt
                if to_save:
                    torch.save(generator.state_dict(), os.path.join(saving_path, f"best-G.pth"))
                    torch.save(discriminator.state_dict(), os.path.join(saving_path, f"best-D.pth"))
                    print("best models saved")

        if epoch % args.saving_interval == 0 and to_save:
            torch.save(generator.state_dict(), os.path.join(saving_path, f"latest-G.pth"))
            torch.save(discriminator.state_dict(), os.path.join(saving_path, f"latest-D.pth"))
            print("latest models saved")

        if args.local_rank == 0:
            print(f"best acc G (t): {best_acc_g_tgt:.3f}" +
                  (f" | saving path: {saving_path}\n" if to_save else "\n") +
                  f"dataset: {args.dataset} (src{len(os.listdir(train_src_path))}-tgt{len(os.listdir(train_tgt_path))})"
                  f" | mode: {mode} | batch size: {batch_size} | epochs: {epochs}"
                  f" | G src steps: {g_src_steps}{' (reducing)' if args.reducing and g_src_steps > 0 else ''}" +
                  (f" | G few-shot tgt steps: {g_fs_tgt_steps}" if is_few_shot else "") +
                  f" | G steps: {g_steps} | D steps: {d_steps}" +
                  (f" | no G weights\n" if args.g_weights == "" else "\n") +
                  (f"remarks: {args.remarks}\n\n" if args.remarks is not None else "\n\n"))

    dist.destroy_process_group()


if __name__ == '__main__':
    date_time = datetime.datetime.now()
    parser = argparse.ArgumentParser()

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--g-src', action="store_true")
    parser.add_argument('--g-src-steps', type=int, default=2)
    parser.add_argument('--reducing', action="store_true")

    parser.add_argument('--g-fs-tgt-steps', type=int, default=0)

    parser.add_argument('--g-steps', type=int, default=1)
    parser.add_argument('--d-steps', type=int, default=1)

    parser.add_argument('--lr-g', type=float, default=0.000002)
    parser.add_argument('--lr-d', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--eval-interval', type=int, default=1)

    # datasets
    parser.add_argument('--data-root', type=str, default='./datasets/datasets')
    parser.add_argument('--dataset', type=str, default='mini-imagenet')
    parser.add_argument('--sub', type=str, default='src50-tgt50')

    # weights
    parser.add_argument('--weights-root', type=str, default='./weights')
    parser.add_argument('--g-weights', type=str, help='initial g_weights path')
    parser.add_argument('--d-weights', type=str, default='', help='initial d_weights path')

    # save weights
    parser.add_argument('--do-not-save', action="store_true")
    parser.add_argument('--saving-root', type=str, default='./runs')
    parser.add_argument('--date-time', type=str,
                        default=f'{date_time.month:02d}-{date_time.day:02d}_'
                                f'{date_time.hour:02d}-{date_time.minute:02d}-{date_time.second:02d}')
    parser.add_argument('--saving-interval', type=int, default=5)

    # remarks
    parser.add_argument('--remarks', type=str)

    args = parser.parse_args()

    train(args)
