from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


class WriterHDF5Embedding:
    def __init__(
        self,
        file_name: str,
        group_name: str = "embeddings",
        ids_name: str = "ids",
        compression: str | None = "gzip",  # "gzip" | "lzf" | None
        compression_opts: int | None = 4,  # gzip level
        chunk_rows: int | None = None,  # None = the batch size of the first write
    ):
        self._group_name = group_name
        self._ids_name = ids_name
        self._compression = compression
        self._compression_opts = compression_opts if compression == "gzip" else None
        self._chunk_rows = int(chunk_rows) if chunk_rows else None

        self._filepath = Path(file_name)
        self._h5: h5py.File | None = None
        self._grp = None
        self._ds_ids = None
        self._n = 0

    def _stored_row_count(self) -> int:
        if not self._filepath.exists():
            return 0
        with h5py.File(str(self._filepath), "r") as f:
            if self._ids_name not in f:
                return 0
            return int(f[self._ids_name].shape[0])

    def check_target(self) -> None:
        """Raise if the file already holds rows."""
        stored = self._stored_row_count()
        if stored:
            raise FileExistsError(
                f"{self._filepath} already holds {stored} rows, and mixing two runs "
                f"in one file is not something this writer does. Point run_directory= "
                f"at a folder of its own, or clear that one out."
            )

    def open(self):
        if self._h5 is not None:
            return

        self.check_target()

        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(str(self._filepath), "a")
        self._grp = self._h5.require_group(self._group_name)

        if self._ids_name in self._h5:
            self._ds_ids = self._h5[self._ids_name]
            self._n = int(self._ds_ids.shape[0])
        else:
            self._ds_ids = self._h5.create_dataset(
                self._ids_name,
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
                chunks=True,
            )
            self._n = 0

    def _tensor_to_numpy(self, name: str, t: torch.Tensor) -> np.ndarray:
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Embedding '{name}': expected torch.Tensor, got {type(t)}")

        t = t.detach().to("cpu").contiguous()
        if t.is_floating_point() and t.element_size() < 4:
            t = t.to(torch.float32)

        arr = t.numpy()
        if arr.ndim != 2:
            raise ValueError(
                f"Embedding '{name}' must have shape [B, D], got {tuple(arr.shape)}"
            )
        return arr

    def _get_or_create_dataset(
        self, name: str, dim: int, dtype: np.dtype, rows: int
    ) -> h5py.Dataset:
        if name in self._grp:
            ds = self._grp[name]
            if int(ds.shape[1]) != dim:
                raise ValueError(
                    f"Embedding '{name}' dimension mismatch: file has D={ds.shape[1]}, new batch has D={dim}"
                )
            if ds.dtype != dtype:
                raise ValueError(
                    f"Embedding '{name}' dtype mismatch: file has {ds.dtype}, new batch has {dtype}"
                )
            return ds

        return self._grp.create_dataset(
            name,
            shape=(0, dim),
            maxshape=(None, dim),
            dtype=dtype,
            chunks=(self._chunk_rows or rows, dim),
            compression=self._compression,
            compression_opts=self._compression_opts,
        )

    def write(self, ids: Sequence[Any], embeddings: dict[str, torch.Tensor]):
        """Append one batch: ids of length B, embeddings name -> Tensor[B, D]."""
        if not isinstance(embeddings, dict) or len(embeddings) == 0:
            raise ValueError(
                "embeddings must be a non-empty dict: name -> Tensor[B, D]"
            )

        ids_list = [str(x) for x in ids]
        arrays = {
            name: self._tensor_to_numpy(name, t) for name, t in embeddings.items()
        }

        batch_sizes = {arr.shape[0] for arr in arrays.values()} | {len(ids_list)}
        if len(batch_sizes) != 1:
            sizes = {name: arr.shape[0] for name, arr in arrays.items()}
            raise ValueError(
                f"Inconsistent batch sizes: ids={len(ids_list)}, embeddings={sizes}"
            )

        B = len(ids_list)
        if B == 0:
            return

        self.open()
        old_n, new_n = self._n, self._n + B

        try:
            self._ds_ids.resize((new_n,))
            self._ds_ids[old_n:new_n] = ids_list

            for name, arr in arrays.items():
                ds = self._get_or_create_dataset(name, int(arr.shape[1]), arr.dtype, B)
                ds.resize((new_n, int(arr.shape[1])))
                ds[old_n:new_n] = arr

            self._n = new_n

        except Exception:
            self._ds_ids.resize((old_n,))
            for name in arrays:
                if name in self._grp:
                    ds = self._grp[name]
                    ds.resize((old_n, ds.shape[1]))
            raise

    def __call__(self, batch: dict[str, Any]):
        if not batch:
            return
        ids = batch.get("id", None)
        embeddings = batch.get("embeddings", None)
        if ids is None or embeddings is None:
            raise KeyError(
                "WriterHDF5Embedding expects batch keys: 'id' and 'embeddings' dict."
            )
        self.write(ids=ids, embeddings=embeddings)

    def flush(self):
        if self._h5 is not None:
            self._h5.flush()

    def close(self):
        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None
            self._grp = None
            self._ds_ids = None
