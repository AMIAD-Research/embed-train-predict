import logging
import time
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import lightning.pytorch as pl
import torch

logger = logging.getLogger(__name__)


class PostProcessStep(pl.callbacks.Callback):
    """
    Gathers the embed-phase outputs across ranks and writes them on rank 0.

    Works with DDP / DDPSpawn and DeepSpeedStrategy (single or multi node).

    With `use_distributed_sampler: true`, torch's DistributedSampler pads the
    dataset so that every rank runs the same number of steps (required by
    ZeRO-3 collectives). The padded duplicates are dropped here before writing,
    keyed on the 'id' field (ids must therefore be unique across the dataset).

    `gather_keys` restricts what crosses the ranks: everything the step returns
    but the writer ignores (token ids, masks, raw texts) would otherwise be
    all_gathered every batch, the non-tensors through pickle.

    `writer` is anything callable as writer(batch_dict) that also has
    check_target(), open(), flush() and close() methods,
    etp.embedder.writers.WriterHDF5Embedding being the one this pipeline ships.
    """

    def __init__(
        self,
        writer,
        gather_keys: Sequence[str] | None = None,
        flush_every_n_steps: int = 1,
    ):
        super().__init__()
        self._writer = writer
        self._gather_keys = tuple(gather_keys) if gather_keys is not None else None
        self._flush_every_n_steps = max(1, int(flush_every_n_steps))
        self._seen_ids = set()
        self._write_seconds = 0.0
        self._rows_written = 0

    def _dist_on(self, trainer) -> bool:
        return (
            getattr(trainer, "world_size", 1) > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )

    def _is_dict_of_tensors(self, v):
        return (
            isinstance(v, dict)
            and len(v) > 0
            and all(isinstance(k, str) for k in v)
            and all(isinstance(t, torch.Tensor) for t in v.values())
        )

    def _gather_object_list(self, obj: Any, trainer) -> list[Any]:
        """Gather arbitrary python objects from all ranks (rank order)."""
        if not self._dist_on(trainer):
            return [obj]

        gathered = [None for _ in range(trainer.world_size)]
        torch.distributed.all_gather_object(gathered, obj)
        return gathered

    def _merge_gathered_dicts(
        self, dicts_per_rank: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Merge list of dicts (one per rank) into a single dict of concatenated lists."""
        merged: dict[str, list[Any]] = {}
        for d in dicts_per_rank:
            for k, v in d.items():
                merged.setdefault(k, [])
                if isinstance(v, (list, tuple)):
                    merged[k].extend(list(v))
                else:
                    merged[k].append(v)
        return merged

    def _gather_outputs(
        self, trainer, pl_module, outputs: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Gather outputs across ranks. Returns:
          - dict on rank 0
          - None on non-zero ranks (so they do nothing)
        """
        outputs = outputs if isinstance(outputs, dict) else {}
        if self._gather_keys is not None:
            outputs = {k: v for k, v in outputs.items() if k in self._gather_keys}

        if not self._dist_on(trainer):
            return outputs

        tensor_part: dict[str, torch.Tensor] = {}
        object_part: dict[str, Any] = {}

        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                tensor_part[k] = v
            elif self._is_dict_of_tensors(v):
                # flatten dict[str, Tensor] into tensor_part
                for kk, tt in v.items():
                    tensor_part[f"{k}::{kk}"] = tt
            else:
                object_part[k] = v

        gathered_tensors: dict[str, Any] = {}
        for k, t in tensor_part.items():
            gt = pl_module.all_gather(t)
            # collapse the world dimension if present: [W,B,...] -> [W*B,...]
            if (
                isinstance(gt, torch.Tensor)
                and gt.ndim >= 2
                and gt.shape[0] == trainer.world_size
            ):
                gt = gt.reshape(-1, *gt.shape[2:])
            elif (
                isinstance(gt, torch.Tensor)
                and gt.ndim == 1
                and gt.shape[0] == trainer.world_size
            ):
                gt = gt.reshape(-1)
            gathered_tensors[k] = gt

        # Make non-tensors *lists* per rank so merge is clean.
        obj_for_gather = {}
        for k, v in object_part.items():
            obj_for_gather[k] = list(v) if isinstance(v, (list, tuple)) else [v]

        dicts_per_rank = self._gather_object_list(obj_for_gather, trainer)

        if trainer.global_rank != 0:
            return None

        gathered = self._merge_gathered_dicts(dicts_per_rank)
        gathered.update(gathered_tensors)

        # rebuild dict[str, Tensor] that were flattened as "key::subkey"
        nested = {}
        for k, v in list(gathered.items()):
            if isinstance(k, str) and "::" in k and isinstance(v, torch.Tensor):
                root, leaf = k.split("::", 1)
                nested.setdefault(root, {})[leaf] = gathered.pop(k)
        gathered.update(nested)

        return gathered

    def _select_rows(self, value, keep: list[int], num_rows: int):
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and value.shape[0] == num_rows
        ):
            return value[keep]
        if isinstance(value, dict):
            return {k: self._select_rows(v, keep, num_rows) for k, v in value.items()}
        if isinstance(value, (list, tuple)) and len(value) == num_rows:
            return [value[i] for i in keep]
        return value

    def _drop_duplicates(self, gathered: dict[str, Any]) -> dict[str, Any] | None:
        """Drop rows whose id was already written (DistributedSampler padding)."""
        ids = gathered.get("id")
        if ids is None:
            return gathered

        keep = []
        for i, sample_id in enumerate(ids):
            if sample_id not in self._seen_ids:
                self._seen_ids.add(sample_id)
                keep.append(i)

        if len(keep) == len(ids):
            return gathered
        if not keep:
            return None
        return {k: self._select_rows(v, keep, len(ids)) for k, v in gathered.items()}

    def _on_batch_end(self, trainer, pl_module, outputs, batch_idx):
        gathered = self._gather_outputs(trainer, pl_module, outputs)
        if gathered is not None and trainer.global_rank == 0:
            gathered = self._drop_duplicates(gathered)
            started = time.monotonic()
            if gathered is not None:
                self._writer(gathered)
                self._rows_written += len(gathered.get("id", ()))
            if (batch_idx + 1) % self._flush_every_n_steps == 0:
                self._writer.flush()
            self._write_seconds += time.monotonic() - started

    def _on_end(self, trainer):
        if trainer.global_rank == 0:
            started = time.monotonic()
            self._writer.open()
            self._writer.close()
            self._write_seconds += time.monotonic() - started
            self._log_write_cost()

    def _log_write_cost(self) -> None:
        per_row = (
            self._write_seconds / self._rows_written if self._rows_written else 0.0
        )
        logger.info(
            f"Writer: {self._rows_written} rows in {timedelta(seconds=round(self._write_seconds))} "
            f"({per_row * 1000:.0f} ms/row, embed-loop time outside the writer is the rest)"
        )

    def setup(self, trainer, pl_module, stage=None):
        if stage == "test":
            self._writer.check_target()

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._on_batch_end(trainer, pl_module, outputs, batch_idx)

    def on_test_end(self, trainer, pl_module):
        self._on_end(trainer)
