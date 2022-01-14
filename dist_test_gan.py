import argparse
import os
import random

import torch
import torch.distributed as dist
import torch.utils.data as Data
from sklearn import metrics
from torch import nn
from tqdm import tqdm

from configs.transforms import transforms_dict
from datasets.folder import ImageFolder, make_dataset, IMG_EXTENSIONS, pil_loader
from utils.gan import get_model, load_weights

torch.set_printoptions(profile="full")


@torch.no_grad()
def dist_test(args):
    batch_size = args.batch_size
    if args.g_weights is None:
        args.g_weights = os.path.join(args.weights_root, args.id, "best-G.pth")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist.get_world_size()

    if args.dataset == "mini-imagenet":
        s = "src50-tgt50"
    elif args.dataset in ["cifar10", "mnist", "stl10"]:
        s = "src5-tgt5"
    elif args.dataset == "pets":
        s = "src19-tgt18"
    else:
        raise Exception

    test_tgt_path = os.path.join(args.data_root, args.dataset, "test_tgt")
    test_tgt_repr_path = os.path.join(args.data_root, args.dataset, f"{s}/train_tgt")

    if args.local_rank == 0:
        print(f"world size: {world_size}\n"
              f"device: {device}\n"
              f"Using {num_workers} dataloader workers every process\n\n"
              f" test tgt path: {test_tgt_path}\n"
              f"g-weights path: {args.g_weights}\n")

    transform = transforms_dict[args.dataset]["val"]
    test_tgt_dataset_g = ImageFolder(root=test_tgt_path, transform=transform)
    test_tgt_sampler_g = Data.distributed.DistributedSampler(dataset=test_tgt_dataset_g, shuffle=False)
    test_tgt_loader_g = Data.DataLoader(dataset=test_tgt_dataset_g,
                                        sampler=test_tgt_sampler_g,
                                        batch_size=batch_size,
                                        pin_memory=True,
                                        num_workers=num_workers)

    class_to_images, _, num_images = make_dataset(directory=test_tgt_repr_path,
                                                  class_to_idx=test_tgt_dataset_g.class_to_idx,
                                                  extensions=IMG_EXTENSIONS)
    num_classes = len(class_to_images)
    class_to_idx = test_tgt_dataset_g.class_to_idx

    generator = get_model(dataset=args.dataset, mode='G', device=device)
    load_weights(model=generator, weights=args.g_weights, device=device)
    generator = nn.parallel.DistributedDataParallel(generator, device_ids=[args.local_rank])
    generator.eval()

    # test G with tgt
    test_tgt_loader_g_ = tqdm(test_tgt_loader_g) if args.local_rank == 0 else test_tgt_loader_g
    pred_all, tgt_labels_all = [], []
    acc = 0.

    for tgt_images, tgt_labels in test_tgt_loader_g_:
        tgt_images = tgt_images.to(device)
        tgt_labels = tgt_labels.to(device)
        b = tgt_labels.shape[0]  # B

        pred = torch.zeros([b, num_classes]).to(device)  # [B, num_classes]

        for cls in class_to_images.keys():
            cls_idx = class_to_idx[cls]

            if args.local_rank == 0:
                print(cls, cls_idx, end=" ")

            repr = random.sample(class_to_images[cls], args.num_shot)

            for img in repr:
                repr_images = transform(pil_loader(img)).to(device).unsqueeze(0).expand(b, -1, -1, -1)  # [B, 3, H, W]
                tgt_pairs = torch.stack([tgt_images, repr_images], dim=1)  # [B, 2, 3, H, W]

                g_out = generator(tgt_pairs)  # [B, 1]
                g_out = (g_out + (1 - 0.5)).int()

                for i in range(b):
                    pred[i][cls_idx] += g_out[i][0]

        pred = torch.argmax(pred, dim=1)  # [B]

        pred_list = [torch.zeros(tgt_labels.shape[0], dtype=torch.int64).to(device) for _ in range(world_size)]
        tgt_labels_list = [torch.zeros(tgt_labels.shape[0], dtype=torch.int64).to(device) for _ in range(world_size)]
        dist.all_gather(pred_list, pred)
        dist.all_gather(tgt_labels_list, tgt_labels)
        for i in pred_list:
            pred_all.extend(i.cpu().numpy().tolist())
        for i in tgt_labels_list:
            tgt_labels_all.extend(i.cpu().numpy().tolist())

        if args.local_rank == 0:
            print(f"\n  pred: {pred.cpu().numpy().tolist()}\n"
                  f"labels: {tgt_labels.cpu().numpy().tolist()}\n")

        acc = metrics.accuracy_score(tgt_labels_all, pred_all)
        macro_f1 = metrics.f1_score(tgt_labels_all, pred_all, average='macro')

        if args.local_rank == 0:
            test_tgt_loader_g_.desc = f"acc G (t): {acc:.4f} | f1-score G (t): {macro_f1:.4f}"

    if args.local_rank == 0:
        print()

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-shot', type=int, default=1)

    # datasets
    parser.add_argument('--data-root', type=str, default='./datasets/datasets')
    parser.add_argument('--dataset', type=str)

    # weights
    parser.add_argument('--weights-root', type=str, default='./runs')
    parser.add_argument('--id', type=str, help='checkpoint id')
    parser.add_argument('--g-weights', type=str, help='initial g_weights path')

    args = parser.parse_args()

    dist_test(args)
