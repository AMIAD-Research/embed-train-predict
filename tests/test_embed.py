"""CPU-only tests of phase 1 (embed): config composition, data pipeline, writer, dedup.

No GPU, no network, no HF model download (stub tokenizer).
"""

import h5py
import hydra
import numpy
import pandas
import pytest
import stubs
import torch
from conftest import CONFIG_DIR, run_cli
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from etp.cli.embed import HYDRA_VERSION_BASE


class StubTrainer:
    """A single-process trainer, enough for the writer and callback hooks."""

    world_size = 1
    global_rank = 0


def _compose(corpus_dir, out_dir, extra=()):
    overrides = [
        "corpus=clc-fce",
        f"run_directory={out_dir}/run",
        f"corpus_directory={corpus_dir}",
        "model_name_or_path=/fake/hf/model",
        "datamodule.test_dataloader_cfg.batch_size=2",
        "datamodule.test_dataloader_cfg.num_workers=0",
        "datamodule.test_dataloader_cfg.persistent_workers=false",
        "datamodule.test_dataloader_cfg.pin_memory=false",
        "datamodule.collate_fn_cfg.processor={_target_: stubs.StubProcessor}",
        "datamodule.collate_fn_cfg.config={_target_: stubs.StubConfig}",
        *extra,
    ]
    # the same semantics run_embed() composes under, not Hydra's 1.1 fallback
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR), version_base=HYDRA_VERSION_BASE
    ):
        return compose(config_name="config", overrides=overrides)


def test_corpus_loading(corpus_dir):
    from etp.embedder.dataset import TableDataset
    from etp.utils import concat_corpora, load_corpus_table

    t = load_corpus_table(
        str(corpus_dir / "clc-fce"),
        "list/clc-fce-test.csv",
        constants={"corpus_name": "clc-fce", "split": "test"},
    )
    assert len(t) == 3  # 4 rows in the annotation csv, 3 listed
    assert set(t.columns) == {
        "id",
        "language_l1",
        "text",
        "extra",
        "corpus_name",
        "split",
    }

    ds = TableDataset(concat_corpora([t, t]), columns=["id", "language_l1", "text"])
    assert len(ds) == 6
    assert set(ds[0].keys()) == {"id", "language_l1", "text"}


def test_an_id_listed_but_absent_from_the_annotation_is_refused(corpus_dir):
    from etp.utils import load_corpus_table

    list_file = corpus_dir / "clc-fce" / "list" / "clc-fce-test.csv"
    rows = pandas.read_csv(list_file)
    rows.loc[0, "id"] = "clc-fce-test-does-not-exist"
    rows.to_csv(list_file, index=False)

    with pytest.raises(ValueError, match="absent from"):
        load_corpus_table(str(corpus_dir / "clc-fce"), "list/clc-fce-test.csv")


def test_embed_config_and_dataloader(corpus_dir, tmp_path):
    cfg = _compose(corpus_dir, tmp_path / "out")

    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["model"]["cfg"]["pretrained_model"]["_target_"].endswith(
        "AutoModel.from_pretrained"
    )
    assert resolved["model"]["cfg"]["pretrained_model"]["dtype"] == "bfloat16"
    assert resolved["datamodule"]["collate_fn_cfg"]["prompt"] is None

    data = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
    data.setup("test")
    assert len(data.test_dataset) == 9  # 3 splits x 3 listed rows

    batch = next(iter(data.test_dataloader()))
    assert set(batch.keys()) == {"id", "source_text", "language_l1", "ids", "mask"}
    assert batch["ids"].shape[0] == 2 and batch["mask"].shape == batch["ids"].shape


def test_limit_truncates_dataset(corpus_dir, tmp_path):
    # a fraction of the corpus, e.g. for a quick CPU run
    cfg = _compose(corpus_dir, tmp_path / "out", extra=["dataset.limit=5"])
    data = hydra.utils.instantiate(cfg.datamodule, _recursive_=False)
    data.setup("test")
    assert len(data.test_dataset) == 5


def test_prompt_conditioning_collate(corpus_dir, tmp_path):
    # a prompt is optional and, when set, conditions the text sent to the encoder
    cfg = _compose(corpus_dir, tmp_path / "out", extra=["prompt=instruction"])
    assert cfg.prompt is not None and "system" in cfg.prompt

    from etp.embedder.collate import text_collate

    items = [
        {"id": "a", "language_l1": "fra", "text": "a first learner text"},
        {"id": "b", "language_l1": "deu", "text": "a second learner text"},
    ]
    out = text_collate(
        items,
        processor={"_target_": "stubs.StubProcessor"},
        config={"_target_": "stubs.StubConfig"},
        prompt=cfg.prompt,
    )
    # the stub chat template concatenates the message contents, so the system
    # prompt must appear at the start of each conditioned text
    assert "prompt_text" in out
    assert out["prompt_text"][0].startswith(cfg.prompt.system.split("\n")[0])
    assert out["ids"].shape[0] == 2


@pytest.mark.parametrize(
    "declared, max_position_embeddings, expected",
    [
        (512, 514, 512),
        (10**9, 32768, 32768),
    ],
)
def test_the_config_caps_the_truncation_but_never_raises_it(
    declared, max_position_embeddings, expected
):
    from etp.embedder.collate import text_collate

    class Config:
        pass

    config = Config()
    config.max_position_embeddings = max_position_embeddings

    processor = stubs.StubProcessor()
    processor.model_max_length = declared

    text_collate([{"id": "a", "text": "three plain words"}], processor, config)
    assert processor.model_max_length == expected


def test_deepspeed_strategy_composes(corpus_dir, tmp_path):
    # the ZeRO-3 strategy group must compose and resolve without deepspeed installed
    cfg = _compose(corpus_dir, tmp_path / "out", extra=["strategy=deepspeed"])
    strategy = OmegaConf.to_container(cfg, resolve=True)["trainer"]["strategy"]
    assert strategy["_target_"].endswith("DeepSpeedStrategy")
    assert strategy["config"]["zero_optimization"]["stage"] == 3


def test_pretrained_model_loads_onto_the_rank_device():
    # the checkpoint must be read straight onto the device of the rank loading it,
    # otherwise the weights stay mmap views that the strategy faults in one
    # parameter at a time after setup() (hours for a 141 GB model)
    from etp.embedder.model import EmbedModule

    class StubStrategy:  # not named DeepSpeedStrategy: the from_pretrained branch
        def __init__(self, device):
            self.root_device = torch.device(device)

    class StubTrainer:
        def __init__(self, device):
            self.strategy = StubStrategy(device)

    cfg = OmegaConf.create(
        {"pretrained_model": {"_target_": "stubs.StubPretrainedModel"}}
    )

    for device in ("cuda:3", "cpu"):  # rank 3 of a multi-GPU node, then a CPU run
        module = EmbedModule(cfg)
        module._trainer = StubTrainer(device)
        module.setup("test")
        assert module.model.kwargs["device_map"] == device


def test_the_weights_are_released_before_the_trainer_moves_them_to_host_ram():
    # Trainer._teardown() calls Strategy.teardown() first, and that does
    # lightning_module.cpu(): every parameter back over the bus, into a job that
    # never sized its host memory for a model that only fits because it has the
    # GPU to itself.
    from etp.embedder.model import EmbedModule

    module = EmbedModule(OmegaConf.create({}))
    module.model = torch.nn.Linear(4, 4)
    assert list(module.children())  # registered: .cpu() would walk it

    module.on_test_end()

    assert module.model is None
    assert list(module.children()) == []
    module.cpu()  # the copy the strategy makes on the way out, a no-op now


def test_embed_module_masked_mean_pools_hidden_states():
    # test_step returns the mask-weighted mean over the sequence, so padding
    # positions are excluded from the embedding
    from etp.embedder.model import EmbedModule

    module = EmbedModule(
        OmegaConf.create({"processor": {"_target_": "stubs.StubProcessor"}})
    )

    B, T, D = 2, 3, 4
    h0 = torch.arange(B * T * D, dtype=torch.float32).reshape(B, T, D)
    h1 = h0 + 100.0

    class StubOutput:
        hidden_states = (h0, h1)

    class StubModel:
        def __call__(self, input_ids, attention_mask, output_hidden_states):
            assert output_hidden_states is True
            return StubOutput()

    module.model = StubModel()  # bypass setup(): no real model needed

    # sample 0: all 3 tokens real; sample 1: first position is left padding
    mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    out = module.test_step(
        {"ids": torch.zeros(B, T, dtype=torch.long), "mask": mask}, 0
    )

    assert set(out["embeddings"].keys()) == {"layer_0", "layer_1"}
    for name, h in [("layer_0", h0), ("layer_1", h1)]:
        # independent reference: plain mean over the real positions only
        expected = torch.stack([h[0].mean(dim=0), h[1, 1:].mean(dim=0)])
        assert out["embeddings"][name].shape == (B, D)
        assert torch.allclose(out["embeddings"][name], expected)


def test_merge_gathered_dicts_concatenates_per_rank():
    # the object-merge step behind the multi-rank gather: one dict per rank -> concat
    from etp.embedder.callbacks import PostProcessStep

    step = PostProcessStep(writer=None)
    merged = step._merge_gathered_dicts([{"id": ["a", "b"]}, {"id": ["c", "a"]}])
    assert merged["id"] == ["a", "b", "c", "a"]


def test_dedup_drops_padded_duplicates_across_ranks():
    # as assembled on rank 0 after gathering 2 ranks: rank 1's trailing 'a' is
    # DistributedSampler padding and must be dropped, its tensor row with it
    from etp.embedder.callbacks import PostProcessStep

    step = PostProcessStep(writer=None)
    emb = torch.tensor([[0.0], [1.0], [2.0], [0.0]])  # rows for a, b, c, a(pad)
    gathered = {"id": ["a", "b", "c", "a"], "embeddings": {"layer_0": emb}}

    kept = step._drop_duplicates(gathered)
    assert kept["id"] == ["a", "b", "c"]
    assert torch.equal(kept["embeddings"]["layer_0"], emb[[0, 1, 2]])


def test_writer_and_deduplication(corpus_dir, tmp_path):
    cfg = _compose(corpus_dir, tmp_path / "out")

    cb = hydra.utils.instantiate(cfg.embed_writer)
    emb_file = (
        cfg.embed_writer.writer.file_name
    )  # resolved <run_directory>/embeddings.h5

    emb1 = {"layer_0": torch.randn(2, 8)}
    emb2 = {"layer_0": torch.randn(2, 8)}
    batch1 = {"id": ["a", "b"], "language_l1": ["fra", "deu"], "embeddings": emb1}
    batch2 = {"id": ["b", "c"], "language_l1": ["deu", "spa"], "embeddings": emb2}

    cb.on_test_batch_end(StubTrainer(), None, dict(batch1), None, 0)
    cb.on_test_batch_end(StubTrainer(), None, dict(batch1), None, 1)  # full duplicate
    cb.on_test_batch_end(
        StubTrainer(), None, dict(batch2), None, 2
    )  # partial duplicate
    cb.on_test_end(StubTrainer(), None)

    with h5py.File(emb_file) as f:
        assert [x.decode() for x in f["ids"][:]] == ["a", "b", "c"]

    from etp.utils import load_embeddings

    stored = load_embeddings(emb_file, "layer_0")
    assert torch.allclose(torch.tensor(stored.loc["a"].to_numpy()), emb1["layer_0"][0])
    # row for 'c' must be batch2's second row (dedup kept the right tensor row)
    assert torch.allclose(torch.tensor(stored.loc["c"].to_numpy()), emb2["layer_0"][1])


def test_chunks_cover_the_rows_one_write_appends(tmp_path):
    # a chunk is the unit the filter pipeline works on, so a write covering a
    # fraction of one has to read it back, decompress it, patch it and
    # recompress it. h5py's `chunks=True` heuristic knows nothing about the
    # write pattern and picks (32, 512) for an (N, 8192) dataset, spreading one
    # row over 16 chunks and, on a parallel filesystem, over 16 writes
    from etp.embedder.writers import WriterHDF5Embedding

    path = tmp_path / "embeddings.h5"
    writer = WriterHDF5Embedding(file_name=str(path))
    writer.write(ids=["a", "b"], embeddings={"layer_0": torch.randn(2, 8192)})
    writer.close()

    with h5py.File(path) as f:
        assert f["embeddings/layer_0"].chunks == (2, 8192)
        # variable-length strings keep their characters in the file's global
        # heap, out of reach of the filter, so compressing that dataset costs a
        # rewrite of the chunk per append and saves nothing
        assert f["ids"].compression is None


def test_chunk_rows_overrides_the_write_granularity(tmp_path):
    # for a caller that appends in a different granularity than it batches
    from etp.embedder.writers import WriterHDF5Embedding

    path = tmp_path / "embeddings.h5"
    writer = WriterHDF5Embedding(file_name=str(path), chunk_rows=64)
    writer.write(ids=["a"], embeddings={"layer_0": torch.randn(1, 8)})
    writer.close()

    with h5py.File(path) as f:
        assert f["embeddings/layer_0"].chunks == (64, 8)


def test_a_populated_file_is_refused_before_the_weights_are_loaded(tmp_path):
    # Lightning runs the callback setup hooks before the module's, so the run is
    # stopped there. Checking at the first write instead spends the weight
    # loading first, an hour of the allocation on a 405B checkpoint.
    from etp.embedder.callbacks import PostProcessStep
    from etp.embedder.writers import WriterHDF5Embedding

    path = tmp_path / "embeddings.h5"
    seed = WriterHDF5Embedding(file_name=str(path))
    seed.write(["a"], {"layer_0": torch.zeros(1, 8)})
    seed.close()

    step = PostProcessStep(writer=WriterHDF5Embedding(file_name=str(path)))
    with pytest.raises(FileExistsError, match="already holds 1 rows"):
        step.setup(StubTrainer(), None, "test")

    # there is no flag to force it through, and the rows of the first run are left
    # untouched on the way out
    with h5py.File(path) as f:
        assert f["ids"].shape == (1,)


def test_a_run_that_writes_nothing_still_leaves_the_file(tmp_path):
    # empty corpus, or dataset.limit=0: `etp train` must fail on "no embedding
    # stored", which needs the file, not on the file missing altogether
    from etp.embedder.callbacks import PostProcessStep
    from etp.embedder.writers import WriterHDF5Embedding
    from etp.utils import available_layers

    path = tmp_path / "nested" / "embeddings.h5"
    step = PostProcessStep(writer=WriterHDF5Embedding(file_name=str(path)))
    step.setup(StubTrainer(), None, "test")
    step.on_test_end(StubTrainer(), None)

    assert path.exists()
    assert available_layers(path) == []

    # and it holds no row, so it does not stand in the way of a rerun
    PostProcessStep(writer=WriterHDF5Embedding(file_name=str(path))).setup(
        StubTrainer(), None, "test"
    )


def test_a_column_the_collate_would_overwrite_is_refused_at_setup():
    # a corpus column named like one of the collate outputs used to be dropped
    # silently. The check belongs to DataModule.setup(), which Lightning runs
    # before the weights: raising from the collate instead means a dataloader
    # worker failing at the first batch, once the checkpoint is sharded.
    import pandas as pd

    from etp.embedder.dataset import DataModule, TableDataset

    data = DataModule(
        collate_fn_cfg={
            "function": "etp.embedder.collate.text_collate",
            "text_key": "text",
        }
    )
    data.test_dataset = TableDataset(
        pd.DataFrame(
            {"id": ["a"], "mask": ["a real column"], "text": ["some text here"]}
        )
    )

    with pytest.raises(ValueError, match="mask"):
        data.setup("test")

    # the text column itself is exempt, it is replaced by the tokenized form,
    # so a corpus whose text column is named `ids` goes through
    exempt = DataModule(
        collate_fn_cfg={
            "function": "etp.embedder.collate.text_collate",
            "text_key": "ids",
        }
    )
    exempt.test_dataset = TableDataset(
        pd.DataFrame({"id": ["a"], "ids": ["some text here"]})
    )
    exempt.setup("test")


def test_bfloat16_embeddings_keep_their_range_on_the_way_to_the_file():
    # bfloat16 has no numpy equivalent, and float16 has the mantissa for a bf16
    # value (10 bits against 8) but not the range: storing through it would send
    # anything past 65504 to inf without a word
    from etp.embedder.writers import WriterHDF5Embedding

    big = torch.tensor([[70000.0, 1.5]], dtype=torch.bfloat16)
    arr = WriterHDF5Embedding(file_name="unused.h5")._tensor_to_numpy("layer_0", big)

    assert arr.dtype == numpy.float32
    assert numpy.isfinite(arr).all() and arr[0, 0] >= 65504


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_every_float_narrower_than_float32_is_widened(dtype):
    # widening bfloat16 alone left float16 reachable, one
    # `model.cfg.config.dtype=float16` away now that the two dtypes are wired
    # together, and the file would have held float16 where the README promises
    # float32
    from etp.embedder.writers import WriterHDF5Embedding

    stored = WriterHDF5Embedding(file_name="unused.h5")._tensor_to_numpy(
        "layer_0", torch.ones(1, 3, dtype=dtype)
    )
    assert stored.dtype == numpy.float32


def test_zero3_slices_tile_the_parameter_the_way_deepspeed_partitions_it():
    # DeepSpeed pads the parameter up to a multiple of the world size and gives
    # every rank a slice of that one size, the last of them running past the real
    # element count. Spreading the remainder over the first ranks instead, as this
    # once did, puts every rank past the first off by an element.
    from etp.embedder.zero3 import _partition_start_len

    class Param:  # 10 elements, ds_tensor of 12 // 4 = 3 on each of 4 ranks
        ds_numel = 10
        shape = (10,)

    param, ds_tensor = Param(), torch.zeros(3)
    slices = [_partition_start_len(param, ds_tensor, 4, rank) for rank in range(4)]
    assert slices == [(0, 3), (3, 3), (6, 3), (9, 1)]

    # the slices cover the parameter exactly once, no gap and no overlap
    covered = [i for start, length in slices for i in range(start, start + length)]
    assert covered == list(range(10))


def test_zero3_slice_of_an_unpartitioned_parameter_is_the_whole_tensor():
    from etp.embedder.zero3 import _partition_start_len

    param = torch.zeros(4, 3)
    assert _partition_start_len(param, param, 1, 0) == (0, 12)


def test_zero3_rejects_a_partitioning_it_does_not_understand():
    from etp.embedder.zero3 import _partition_start_len

    class Param:
        ds_numel = 100
        shape = (100,)

    # 4 ranks x 3 elements cannot hold 100: better a stop than weights loaded at
    # the wrong offsets
    with pytest.raises(RuntimeError, match="equal-slice"):
        _partition_start_len(Param(), torch.zeros(3), 4, 0)

    # and slices too *large* have to go too. Hierarchical ZeRO++ sizes ds_tensor on
    # a subgroup (here 100/4 over a world of 16), so ranks 4..15 would land past the
    # end of the parameter, take the local_len == 0 exit, and keep the uninitialised
    # weights of blank_model while the run reported success
    with pytest.raises(RuntimeError, match="equal-slice"):
        _partition_start_len(Param(), torch.zeros(25), 16, 7)


def test_embed_runs_end_to_end_on_cpu(corpus_dir, tmp_path):
    # the whole phase through the command line, stub tokenizer and stub model:
    # composition, dataloader, forward, gather, writer, down to the h5 on disk
    run_dir = tmp_path / "run"
    r = run_cli(
        "embed",
        "model_name_or_path=/fake/hf/model",
        f"run_directory={run_dir}",
        f"corpus_directory={corpus_dir}",
        "trainer.accelerator=cpu",
        "trainer.precision=32",
        "trainer.strategy._target_=lightning.pytorch.strategies.SingleDeviceStrategy",
        "datamodule.test_dataloader_cfg.num_workers=0",
        "datamodule.test_dataloader_cfg.pin_memory=false",
        "datamodule.collate_fn_cfg.processor={_target_: stubs.StubProcessor}",
        "datamodule.collate_fn_cfg.config={_target_: stubs.StubConfig}",
        "model.cfg.pretrained_model={_target_: stubs.StubEmbedModel}",
    )
    assert r.returncode == 0, r.stdout + r.stderr

    with h5py.File(run_dir / "embeddings.h5") as f:
        # 3 splits x 3 listed rows, ids unique, one dataset per hidden layer
        assert f["ids"].shape == (9,)
        assert sorted(f["embeddings"].keys()) == ["layer_0", "layer_1"]
        assert f["embeddings/layer_0"].shape == (9, 4)
        assert f["embeddings/layer_0"].dtype == numpy.float32


def test_a_second_embed_run_refuses_to_share_the_run_directory(corpus_dir, tmp_path):
    # the guard of WriterHDF5Embedding, reached through the command line: mixing
    # two runs in one file goes unnoticed until the train phase, where the
    # duplicate index breaks the alignment with the labels
    from etp.embedder.writers import WriterHDF5Embedding

    run_dir = tmp_path / "run"
    seed = WriterHDF5Embedding(file_name=str(run_dir / "embeddings.h5"))
    seed.write(["a"], {"layer_0": torch.zeros(1, 4)})
    seed.close()

    r = run_cli(
        "embed",
        "model_name_or_path=/fake/hf/model",
        f"run_directory={run_dir}",
        f"corpus_directory={corpus_dir}",
        "trainer.accelerator=cpu",
        "trainer.precision=32",
        "trainer.strategy._target_=lightning.pytorch.strategies.SingleDeviceStrategy",
        "datamodule.test_dataloader_cfg.num_workers=0",
        "datamodule.test_dataloader_cfg.pin_memory=false",
        "datamodule.collate_fn_cfg.processor={_target_: stubs.StubProcessor}",
        "datamodule.collate_fn_cfg.config={_target_: stubs.StubConfig}",
        "model.cfg.pretrained_model={_target_: stubs.StubEmbedModel}",
    )
    assert r.returncode != 0
    assert "already holds 1 rows" in r.stdout + r.stderr


def test_reserved_keys_names_every_key_the_collate_writes():
    # RESERVED_KEYS is a contract about text_collate, which lives in another
    # module, and nothing but this test ties the two together. Adding an output
    # key to text_collate without adding it here leaves a corpus column of that
    # name silently overwritten, which is the whole bug check_reserved_keys is
    # there to stop.
    from etp.embedder.collate import text_collate
    from etp.embedder.dataset import RESERVED_KEYS

    stub = {
        "processor": {"_target_": "stubs.StubProcessor"},
        "config": {"_target_": "stubs.StubConfig"},
    }
    row = [{"id": "a", "language_l1": "fra", "text": "some text here"}]
    passthrough = {"id", "language_l1"}

    # prompt_text only appears when a prompt is configured, so both paths count
    written = (set(text_collate(row, **stub)) - passthrough) | (
        set(text_collate(row, **stub, prompt={"system": "s", "user": "u"}))
        - passthrough
    )

    assert written == set(RESERVED_KEYS)
