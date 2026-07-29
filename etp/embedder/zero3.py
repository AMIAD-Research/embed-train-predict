import json
import logging
import math
import os
from collections import defaultdict

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)


def _iter_sharded_tensors(model_dir: str):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")

    if not os.path.exists(index_path):
        logger.info(f"Loading shard {os.path.join(model_dir, 'model.safetensors')}")
        with safe_open(
            os.path.join(model_dir, "model.safetensors"), framework="pt"
        ) as f:
            for src_key in f:
                yield src_key, f.get_tensor(src_key)
    else:
        with open(index_path) as f:
            index = json.load(f)

        # metadata aliases are stored as {dst_key: src_key}
        meta = {k: v for k, v in index.get("metadata", {}).items() if k != "total_size"}
        aliases = defaultdict(list)  # src_key -> [dst_keys...]
        for dst, src in meta.items():
            aliases[src].append(dst)

        by_shard = defaultdict(list)  # file -> [keys...]
        for k, shard in index["weight_map"].items():
            by_shard[shard].append(k)

        for shard, keys in by_shard.items():
            logger.info(f"Loading shard {os.path.join(model_dir, shard)}")
            with safe_open(os.path.join(model_dir, shard), framework="pt") as f:
                for src_key in keys:
                    t = f.get_tensor(src_key)
                    yield src_key, t
                    for dst in aliases.get(src_key, []):
                        yield dst, t


def _dp_world_and_rank():
    import deepspeed

    if deepspeed.comm.is_initialized():
        return deepspeed.comm.get_world_size(), deepspeed.comm.get_rank()
    return 1, 0


def _global_numel(p: torch.nn.Parameter) -> int:
    """Logical element count: the unsharded numel under ZeRO-3, p.numel() otherwise."""
    n = getattr(p, "ds_numel", None)
    return int(n) if n is not None else math.prod(p.shape)


def _partition_start_len(param, target, world_size: int, rank: int):
    """Offset and length, in the flat checkpoint tensor, of the slice `rank` owns."""
    total = _global_numel(param)
    if target is param:
        return 0, total

    partition_size = target.numel()

    if not total <= partition_size * world_size < total + world_size:
        raise RuntimeError(
            f"a ZeRO-3 slice of {partition_size} elements on each of {world_size} ranks "
            f"does not tile the {total} elements of the parameter: the partitioning is "
            f"not the equal-slice one this loader assumes"
        )

    start = rank * partition_size
    return start, max(0, min(partition_size, total - start))


@torch.no_grad()
def load_zero3_shard(model, ckpt_dir):
    ws, rk = _dp_world_and_rank()
    name_to_param = dict(model.named_parameters())

    for name, src in _iter_sharded_tensors(ckpt_dir):
        p = name_to_param.get(name)

        if p is None and name.startswith("model."):
            p = name_to_param.get(name[6:])
            if p is not None:
                logger.debug(f"Matched ckpt key '{name}' to model key '{name[6:]}'")

        if p is None:
            continue

        # ZeRO-3 keeps this rank's slice in p.ds_tensor, a plain model in p itself
        ds_tensor = getattr(p, "ds_tensor", None)
        target = ds_tensor if ds_tensor is not None else p
        target_dtype = target.dtype
        dest_device = target.device

        start, local_len = _partition_start_len(p, target, ws, rk)
        if local_len == 0:
            continue

        # only convert the local slice
        flat = src.view(-1)  # mmapped view
        slice_cpu = flat.narrow(0, start, local_len)  # minimal bytes
        if slice_cpu.dtype is not target_dtype:
            slice_cpu = slice_cpu.to(dtype=target_dtype)

        target.view(-1).narrow(0, 0, local_len).copy_(
            slice_cpu.to(dest_device, non_blocking=True), non_blocking=True
        )
