import os

import torch
import torch.distributed as dist
from tqdm import tqdm

import models


def get_model(dataset, mode, device):
    if dataset == "mnist":
        # model = models.cct_net_base_patch4_28(mode=mode).to(device)
        model = models.cct_net_base_patch16_224(mode=mode).to(device)
    elif dataset == "cifar10":
        # model = models.cct_net_base_patch4_32(mode=mode).to(device)
        model = models.cct_net_base_patch16_224(mode=mode).to(device)
    elif dataset == "stl10":
        # model = models.cct_net_base_patch12_96(mode=mode).to(device)
        model = models.cct_net_base_patch16_224(mode=mode).to(device)
    elif dataset == "pets":
        model = models.cct_net_base_patch16_224(mode=mode).to(device)
    elif dataset == "mini-imagenet":
        model = models.cct_net_base_patch16_224(mode=mode).to(device)
    else:
        raise Exception("No twins model corresponds to the dataset.")

    return model


def load_weights(model, weights, device):
    if weights != '':
        assert os.path.exists(weights), f"weights file: '{weights}' not exist."
        weights_dict = torch.load(weights, map_location=device)
        weights_dict = {k.replace('module.', ''): v for k, v in weights_dict.items()}
        print(model.load_state_dict(weights_dict, strict=False))


def update(model, loss, optim):
    model.zero_grad()
    loss.backward()
    optim.step()


def scaling_update(model, loss, optim, scaler):
    model.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()


@torch.no_grad()
def dist_evaluate(generator, discriminator, val_tgt_loader_g, criterion, device, local_rank):
    generator.eval()
    discriminator.eval()

    # evaluate G with tgt
    total, correct = 0, 0
    acc_g_tgt = None

    val_tgt_loader_g_ = tqdm(val_tgt_loader_g) if local_rank == 0 else val_tgt_loader_g

    for i, (tgt_pair, tgt_similarity) in enumerate(val_tgt_loader_g_):
        tgt_pair = tgt_pair.to(device)
        tgt_similarity = tgt_similarity.to(device)

        g_out = generator(tgt_pair)
        g_loss = criterion(g_out, tgt_similarity)

        total_per_batch = torch.IntTensor([tgt_similarity.shape[0]]).to(device)
        dist.all_reduce(total_per_batch, op=dist.ReduceOp.SUM)
        total += total_per_batch.item()

        correct_per_batch = ((g_out + 0.5).int() == tgt_similarity.int()).sum()
        dist.all_reduce(correct_per_batch, op=dist.ReduceOp.SUM)
        correct += correct_per_batch.item()

        acc_g_tgt = correct / total

        if local_rank == 0:
            val_tgt_loader_g_.desc = f"valid* loss G (t): {g_loss:.3f} | acc G (t): {acc_g_tgt:.3f}"

    return acc_g_tgt
