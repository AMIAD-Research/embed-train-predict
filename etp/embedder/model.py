import gc
import logging
import time
from datetime import timedelta

import hydra
import torch
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class EmbedModule(LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self._cfg_model = OmegaConf.create(cfg)
        self.model = None

    def setup(self, stage: str | None = None):
        """Called by Lightning with stage='test'."""
        strategy_name = self.trainer.strategy.__class__.__name__
        logger.info(f"Setting up model with {strategy_name} strategy (stage={stage})")

        if self.model is None:
            if strategy_name == "DeepSpeedStrategy":
                import deepspeed

                from .zero3 import load_zero3_shard

                torch.cuda.set_device(int(self.trainer.local_rank))

                with deepspeed.zero.Init(
                    config_dict_or_path=self.trainer.strategy.config
                ):
                    self.model = hydra.utils.instantiate(self._cfg_model.blank_model)

                load_zero3_shard(self.model, self._cfg_model.checkpoint_load)
            else:
                self.model = hydra.utils.instantiate(
                    self._cfg_model.pretrained_model,
                    device_map=str(self.trainer.strategy.root_device),
                )

        logger.info(self.model)

    def on_test_start(self):
        self._loop_started_at = time.monotonic()

    def on_test_end(self):
        elapsed = time.monotonic() - getattr(self, "_loop_started_at", time.monotonic())
        logger.info(
            f"Embed loop done in {timedelta(seconds=round(elapsed))}, releasing the weights"
        )

        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Weights released, handing back to the trainer")

    def test_step(self, batch, batch_idx):
        result = self.model(
            input_ids=batch["ids"],
            attention_mask=batch["mask"],
            output_hidden_states=True,
        )

        # mask-weighted mean over the sequence
        mask = batch["mask"].unsqueeze(-1)  # [B, T, 1]
        counts = mask.sum(dim=1).clamp(min=1)  # [B, 1], real tokens per sample

        batch["embeddings"] = {
            f"layer_{i}": (hidden * mask.to(hidden.dtype)).sum(dim=1) / counts
            for i, hidden in enumerate(result.hidden_states)
        }

        return batch
