from functools import partial

import hydra
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

RESERVED_KEYS = ("source_text", "prompt_text", "ids", "mask")


class TableDataset(Dataset):
    def __init__(
        self, table, columns: list[str] | None = None, limit: int | None = None
    ):
        table = table[list(columns)] if columns is not None else table
        self._table = table.head(limit) if limit is not None else table

    @property
    def columns(self) -> list[str]:
        """Keys of the dicts __getitem__ returns, i.e. the columns kept."""
        return list(self._table.columns)

    def __len__(self) -> int:
        return len(self._table)

    def __getitem__(self, idx) -> dict:
        return self._table.iloc[int(idx)].to_dict()


class DataModule(LightningDataModule):
    """
    Lightning DataModule driven by Hydra configs.

    Example Hydra config:

    datamodule:
      _target_: etp.embedder.dataset.DataModule
      test_dataset_cfg: ${dataset}
      test_dataloader_cfg:
        batch_size: 2
        num_workers: 8
        pin_memory: true
        persistent_workers: true
      collate_fn_cfg:
        function: etp.embedder.collate.text_collate
        processor: ${model.cfg.processor}
        config: ${model.cfg.config}
        prompt:
    """

    def __init__(
        self,
        test_dataset_cfg=None,
        test_dataloader_cfg=None,
        collate_fn_cfg=None,
    ):
        super().__init__()

        self.test_dataset_cfg = test_dataset_cfg
        self.test_dataloader_cfg = test_dataloader_cfg
        self.collate_fn_cfg = collate_fn_cfg

        self.test_dataset = None

    def setup(self, stage=None):
        if (
            stage in ("test", None)
            and self.test_dataset_cfg is not None
            and self.test_dataset is None
        ):
            self.test_dataset = hydra.utils.instantiate(self.test_dataset_cfg)

        columns = getattr(self.test_dataset, "columns", None)
        if columns is not None and self.collate_fn_cfg is not None:
            self._check_reserved_keys(columns)

    def _check_reserved_keys(self, columns) -> None:
        clash = sorted(
            set(columns)
            & (set(RESERVED_KEYS) - {self.collate_fn_cfg.get("text_key", "text")})
        )
        if clash:
            raise ValueError(
                f"Corpus columns {clash} collide with the keys the collate function writes "
                f"({', '.join(RESERVED_KEYS)}). Rename them in the annotation csv, or drop "
                f"them from `dataset.columns`."
            )

    def _collate_fn(self):
        cfg = self.collate_fn_cfg
        if cfg is None:
            return None
        kwargs = {key: cfg[key] for key in cfg if key != "function"}
        return partial(hydra.utils.get_method(cfg["function"]), **kwargs)

    def _dataloader(self, dataset, loader_cfg):
        if dataset is None:
            raise RuntimeError("Dataset is None. Did you call setup()?")
        if isinstance(loader_cfg, DictConfig):
            loader_cfg = OmegaConf.to_container(loader_cfg, resolve=True)
        return DataLoader(dataset, collate_fn=self._collate_fn(), **(loader_cfg or {}))

    def test_dataloader(self):
        return self._dataloader(self.test_dataset, self.test_dataloader_cfg)
