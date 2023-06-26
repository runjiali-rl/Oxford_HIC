import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from random import random
from model.clipcap import *
from dataset.hic import HumorDataset
from loss.PCloss import PCloss


EPSILON = 1e-9


def train(dataset: HumorDataset, model: ClipCaptionModel, args,
          lr: float = 2e-5, warmup_steps: int = 5000, output_dir: str = "./checkpoints_mlp", output_prefix: str = ""):

    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.cuda()
    local_rank = int(os.environ['LOCAL_RANK'])
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    train_sampler = DistributedSampler(dataset=dataset, shuffle=True)


    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True,
                                    sampler=train_sampler)

    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)



    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        train_dataloader.sampler.set_epoch(epoch)
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for idx, (tokens, mask, prefix) in tqdm(enumerate(train_dataloader)):
            model.zero_grad()
            tokens, mask, prefix = tokens.cuda(), mask.cuda(), prefix.to(dtype=torch.float32).cuda()
            outputs = model(tokens, prefix, mask)
            logits = outputs[:, dataset.prefix_length - 1: -1]
            if args.pcloss:
                loss = PCloss(logits, tokens, use_ce=args.use_ce)
            else:
                loss = nnf.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()

        progress.close()

    model_without_ddp = model.module

    torch.save(model_without_ddp.state_dict(), outdir)

    return model

def init_distributed():
    # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)

    # synchronizes all the threads to reach this point before moving on
    dist.barrier()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/homes/55/runjia/storage/humor_ViT-B_32_single_demo.pkl')
    parser.add_argument('--out_dir', default='./checkpoints_mlp')  
    parser.add_argument('--prefix', default='bokete_prefix', help='prefix for saved filenames')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--prefix_length', type=int, default=10)
    parser.add_argument('--prefix_length_clip', type=int, default=10)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--pcloss', dest='pcloss', action='store_false')

    args = parser.parse_args()

    init_distributed()
    prefix_length = args.prefix_length
    dataset = HumorDataset(args.data, prefix_length, normalize_prefix=args.normalize_prefix)
    prefix_dim = 640 if args.is_rn else 512

    if args.only_prefix:
        model = ClipCaptionPrefix(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=MappingType.MLP)
        print("Train only prefix")
    else:
        model = ClipCaptionModel(prefix_length, clip_length=args.prefix_length_clip, prefix_size=prefix_dim,
                                  num_layers=args.num_layers, mapping_type=MappingType.MLP)
        print("Train both prefix and GPT")
    sys.stdout.flush()
    train(dataset,  model, args, output_dir=args.out_dir, output_prefix=args.prefix,
          lr=args.lr)


if __name__ == '__main__':
    main()