import logging

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything

logger = logging.getLogger(__name__)

HYDRA_VERSION_BASE = "1.3"


def run_embed(
    overrides: list[str] | None = None,
):
    hydra.initialize(config_path="config", version_base=HYDRA_VERSION_BASE)
    cfg = hydra.compose(
        config_name="config", overrides=overrides, return_hydra_config=True
    )
    HydraConfig.instance().set_config(cfg)
    logger.info(cfg)

    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)
    seed_everything(cfg.seed, workers=True)

    data = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    callbacks = [
        hydra.utils.instantiate(cfg.embed_writer),
        hydra.utils.instantiate(cfg.progress_bar),
    ]

    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)
    trainer.test(model, data)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
